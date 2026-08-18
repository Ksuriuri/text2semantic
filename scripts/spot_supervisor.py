#!/usr/bin/env python3
"""Keep a multi-node Spot run alive: recreate preempted nodes, relaunch, repeat.

This runs *off* the training nodes. A per-node supervisor cannot help here --
when a Spot node is preempted the machine itself is gone, so something outside
has to notice, create a replacement and restart the job. The nodes only need to
survive long enough to have mirrored a checkpoint to GCS.

The loop is deliberately small:

    all nodes RUNNING and no job     -> launch on every node
    all nodes RUNNING and job alive  -> keep watching
    any node not RUNNING             -> stop the job, recreate that node, relaunch

Every action is confined to the instance names passed on the command line. The
supervisor never lists the project and acts on what it finds, because a name it
was not given may well be someone else's machine.

The job resumes from the newest complete checkpoint on
--checkpoint_remote_dir, so a restart costs whatever was trained since the last
mirrored save, not the whole run.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time

MISSING = "MISSING"


def run(command, *, check=False, timeout=600):
    return subprocess.run(
        command, check=check, capture_output=True, text=True, timeout=timeout
    )


class Cloud:
    """The gcloud calls, in one place so tests can replace them."""

    def __init__(self, zone, *, dry_run=False, log=print):
        self.zone = zone
        self.dry_run = dry_run
        self.log = log

    def status(self, name):
        result = run(
            [
                "gcloud",
                "compute",
                "instances",
                "describe",
                name,
                f"--zone={self.zone}",
                "--format=value(status)",
            ]
        )
        if result.returncode != 0:
            # describe fails for an instance that a preemption with
            # --instance-termination-action=DELETE has already removed.
            return MISSING
        return result.stdout.strip() or MISSING

    def internal_ip(self, name):
        result = run(
            [
                "gcloud",
                "compute",
                "instances",
                "describe",
                name,
                f"--zone={self.zone}",
                "--format=value(networkInterfaces[0].networkIP)",
            ]
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    def create(self, name, template):
        command = [
            "gcloud",
            "compute",
            "instances",
            "create",
            name,
            f"--zone={self.zone}",
            f"--source-instance-template={template}",
        ]
        self.log(f"create {name}: {shlex.join(command)}")
        if self.dry_run:
            return True
        # No timeout wrapper here on purpose: a create that is killed part way
        # leaves an instance nobody is tracking, which on this machine type is
        # $50/h of nobody's business.
        result = run(command, timeout=None)
        if result.returncode != 0:
            self.log(f"create {name} failed: {result.stderr.strip()}")
        return result.returncode == 0

    def start(self, name):
        command = ["gcloud", "compute", "instances", "start", name, f"--zone={self.zone}"]
        self.log(f"start {name}")
        if self.dry_run:
            return True
        return run(command, timeout=None).returncode == 0

    def ssh(self, name, remote_command, *, background=False):
        if background:
            remote_command = f"nohup {remote_command} >/dev/null 2>&1 & disown"
        command = [
            "gcloud",
            "compute",
            "ssh",
            name,
            f"--zone={self.zone}",
            "--quiet",
            f"--command={remote_command}",
        ]
        if self.dry_run:
            self.log(f"ssh {name}: {remote_command}")
            return 0, "", ""
        result = run(command)
        return result.returncode, result.stdout, result.stderr


def plan(states, job_running):
    """What to do about the current node states.

    Pure so the state machine can be tested without a cloud: `states` maps each
    instance name to its status string.
    """
    broken = sorted(name for name, status in states.items() if status != "RUNNING")
    if broken:
        return {"action": "repair", "nodes": broken}
    if not job_running:
        return {"action": "launch", "nodes": sorted(states)}
    return {"action": "watch", "nodes": []}


class Supervisor:
    def __init__(self, cloud, nodes, *, template, launch_command, unit, log=print):
        self.cloud = cloud
        self.nodes = list(nodes)
        self.template = template
        self.launch_command = launch_command
        self.unit = unit
        self.log = log

    def states(self):
        return {name: self.cloud.status(name) for name in self.nodes}

    def job_running(self):
        """True only if every node still has the training unit active."""
        for name in self.nodes:
            code, stdout, _ = self.cloud.ssh(
                name, f"systemctl is-active {shlex.quote(self.unit)}"
            )
            if code != 0 or stdout.strip() != "active":
                return False
        return True

    def repair(self, broken):
        for name in broken:
            status = self.cloud.status(name)
            if status == MISSING:
                if not self.cloud.create(name, self.template):
                    return False
            elif status == "TERMINATED":
                # A preemption with the default action stops the instance rather
                # than deleting it; the disk and its stale checkpoint survive,
                # which is exactly why resume reads from GCS.
                if not self.cloud.start(name):
                    return False
            else:
                self.log(f"{name} is {status}, waiting for it to settle")
                return False
        return True

    def stop_job(self):
        for name in self.nodes:
            self.cloud.ssh(name, f"sudo systemctl stop {shlex.quote(self.unit)}")

    def launch(self):
        main_ip = self.cloud.internal_ip(self.nodes[0])
        if not main_ip:
            self.log(f"no internal IP for {self.nodes[0]} yet")
            return False
        for rank, name in enumerate(self.nodes):
            remote = (
                f"sudo systemctl reset-failed {shlex.quote(self.unit)} 2>/dev/null; "
                f"sudo systemd-run --unit={shlex.quote(self.unit)} "
                f"--setenv=T2S_MACHINE_RANK={rank} "
                f"--setenv=T2S_MAIN_IP={main_ip} "
                f"--setenv=T2S_NUM_MACHINES={len(self.nodes)} "
                f"{self.launch_command}"
            )
            code, _, stderr = self.cloud.ssh(name, remote)
            if code != 0:
                self.log(f"launch on {name} failed: {stderr.strip()}")
                return False
            self.log(f"launched rank {rank} on {name} (main {main_ip})")
        return True

    def tick(self):
        states = self.states()
        decision = plan(states, self.job_running())
        self.log(f"{json.dumps(states)} -> {decision['action']}")
        if decision["action"] == "repair":
            # Half a job is worse than none: the survivors sit in a collective
            # waiting for ranks that no longer exist.
            self.stop_job()
            self.repair(decision["nodes"])
        elif decision["action"] == "launch":
            self.launch()
        return decision["action"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node",
        action="append",
        required=True,
        dest="nodes",
        help="instance name; repeat per node, rank 0 first",
    )
    parser.add_argument("--zone", required=True)
    parser.add_argument(
        "--instance-template",
        required=True,
        help="template a preempted-and-deleted node is recreated from",
    )
    parser.add_argument(
        "--launch-command",
        required=True,
        help=(
            "command systemd-run executes on each node, e.g. "
            "'/opt/t2s/bin/bash scripts/train_multinode.sh --train_jsonl ...'"
        ),
    )
    parser.add_argument("--unit", default="t2s-train")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--once", action="store_true", help="one tick, for checking a setup"
    )
    args = parser.parse_args(argv)

    cloud = Cloud(args.zone, dry_run=args.dry_run)
    supervisor = Supervisor(
        cloud,
        args.nodes,
        template=args.instance_template,
        launch_command=args.launch_command,
        unit=args.unit,
    )
    print(f"supervising {', '.join(args.nodes)} in {args.zone}", flush=True)
    while True:
        try:
            supervisor.tick()
        except Exception as exc:  # noqa: BLE001 - a transient API error is not a reason to stop
            print(f"tick failed: {type(exc).__name__}: {exc}", flush=True)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Keep a multi-node Spot run alive: recreate preempted nodes, relaunch, repeat.

This runs *off* the training nodes. A per-node supervisor cannot help here --
when a Spot node is preempted the machine itself is gone, so something outside
has to notice, create a replacement and restart the job. The nodes only need to
survive long enough to have mirrored a checkpoint to object storage.

The loop is deliberately small:

    all nodes RUNNING and no job     -> launch on every node
    all nodes RUNNING and job alive  -> keep watching
    any node not RUNNING             -> stop the job, recreate that node, relaunch

Every action is confined to the instance names passed on the command line. The
supervisor never lists the project and acts on what it finds, because a name it
was not given may well be someone else's machine.

Two clouds, one state machine: `GceCloud` speaks gcloud and `Ec2Cloud` speaks
the aws CLI, and both answer the same five questions. The states are reported in
GCE's vocabulary (RUNNING / TERMINATED / MISSING) because that is what `plan`
already reads, so the recovery rules cannot drift between platforms even though
the two providers behave differently: a GCE preemption stops the instance by
default, while an EC2 Spot interruption terminates it, which is why MISSING has
to be as ordinary a state as TERMINATED.

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
# Two instances answering to one node name is not something to repair by making a
# third: it means an earlier create half-succeeded. Reported as a state of its own
# so `plan` calls it broken, `repair` refuses to act, and the log names it.
AMBIGUOUS = "AMBIGUOUS"
# EC2 has its own state vocabulary; everything else passes through upper-cased and
# `repair` waits it out.
_EC2_STATES = {"running": "RUNNING", "stopped": "TERMINATED"}
_EC2_LIVE = "pending,running,shutting-down,stopping,stopped"
# A transient systemd-run unit does not inherit the operator's shell limits, it
# gets the manager's defaults -- 8 MB of locked memory. EFA registers pinned
# memory directly, so that default is the difference between a run and an
# `ibv_reg_mr failed`, and it is also what `train_multinode.sh` refuses to start
# without.
UNIT_PROPERTIES = ("LimitMEMLOCK=infinity", "LimitNOFILE=1048576")


def run(command, *, check=False, timeout=600):
    return subprocess.run(
        command, check=check, capture_output=True, text=True, timeout=timeout
    )


class Cloud:
    """What the supervisor needs from a cloud, and nothing more.

    Five questions -- status, internal IP, create, start, ssh -- so that adding a
    provider cannot add a rule to the state machine. `runner` is injectable so a
    test can assert the exact command without a network.
    """

    def __init__(self, *, dry_run=False, log=print, runner=run):
        self.dry_run = dry_run
        self.log = log
        self.run = runner


class GceCloud(Cloud):
    """The gcloud calls, in one place so tests can replace them."""

    def __init__(self, zone, **kwargs):
        super().__init__(**kwargs)
        self.zone = zone

    def status(self, name):
        result = self.run(
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
        result = self.run(
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
        result = self.run(command, timeout=None)
        if result.returncode != 0:
            self.log(f"create {name} failed: {result.stderr.strip()}")
        return result.returncode == 0

    def start(self, name):
        command = ["gcloud", "compute", "instances", "start", name, f"--zone={self.zone}"]
        self.log(f"start {name}")
        if self.dry_run:
            return True
        return self.run(command, timeout=None).returncode == 0

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
        result = self.run(command)
        return result.returncode, result.stdout, result.stderr


class Ec2Cloud(Cloud):
    """The same five questions against EC2, for p5 Spot capacity.

    Three things differ from GCE and all three are visible below. A node's *name*
    is only a tag, so every lookup is a filtered describe rather than an
    addressable resource. There is no `gcloud compute ssh` equivalent that manages
    keys and firewall rules, so ssh is plain ssh to the address of the moment.
    And a Spot interruption terminates the instance instead of stopping it, so
    `create` from a launch template is the normal repair, not the rare one.
    """

    def __init__(self, region, *, ssh_user="ubuntu", ssh_key=None, public_ip=False, **kwargs):
        super().__init__(**kwargs)
        self.region = region
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.public_ip = public_ip

    def _instances(self, name):
        """Every non-terminated instance tagged with this node name.

        The state filter is server-side because a long Spot run accumulates
        terminated instances under the same name, and they would otherwise
        dominate -- and paginate -- the answer.
        """
        result = self.run(
            [
                "aws",
                "ec2",
                "describe-instances",
                f"--region={self.region}",
                "--filters",
                f"Name=tag:Name,Values={name}",
                f"Name=instance-state-name,Values={_EC2_LIVE}",
                "--output=json",
            ]
        )
        if result.returncode != 0:
            self.log(f"describe {name} failed: {result.stderr.strip()}")
            return []
        payload = json.loads(result.stdout or "{}")
        return [
            instance
            for reservation in payload.get("Reservations", [])
            for instance in reservation.get("Instances", [])
            if instance.get("State", {}).get("Name") != "terminated"
        ]

    def status(self, name):
        live = self._instances(name)
        if not live:
            return MISSING
        if len(live) > 1:
            ids = ", ".join(sorted(i.get("InstanceId", "?") for i in live))
            self.log(f"{name} names {len(live)} live instances ({ids}); not touching it")
            return AMBIGUOUS
        state = live[0].get("State", {}).get("Name", "")
        return _EC2_STATES.get(state, state.upper() or MISSING)

    def internal_ip(self, name):
        """The private address, which is what NCCL rendezvous must use."""
        live = self._instances(name)
        if len(live) != 1:
            return ""
        return live[0].get("PrivateIpAddress", "") or ""

    def _ssh_host(self, name):
        live = self._instances(name)
        if len(live) != 1:
            return ""
        instance = live[0]
        if self.public_ip:
            return instance.get("PublicIpAddress", "") or ""
        return instance.get("PrivateIpAddress", "") or ""

    def create(self, name, template):
        command = [
            "aws",
            "ec2",
            "run-instances",
            f"--region={self.region}",
            "--launch-template",
            f"LaunchTemplateName={template}",
            "--count=1",
            # run-instances replaces the template's instance tags rather than
            # merging, so the Name tag every lookup depends on has to be repeated
            # here even if the template carries one.
            "--tag-specifications",
            f"ResourceType=instance,Tags=[{{Key=Name,Value={name}}}]",
            "--output=json",
        ]
        self.log(f"create {name}: {shlex.join(command)}")
        if self.dry_run:
            return True
        # No timeout wrapper, for the GCE reason: a request killed after the API
        # accepted it leaves a p5 nobody is tracking.
        result = self.run(command, timeout=None)
        if result.returncode != 0:
            self.log(f"create {name} failed: {result.stderr.strip()}")
        return result.returncode == 0

    def start(self, name):
        """Only reachable for a persistent Spot request, which stops on interrupt."""
        live = self._instances(name)
        if len(live) != 1:
            self.log(f"start {name}: expected one instance, found {len(live)}")
            return False
        instance_id = live[0].get("InstanceId", "")
        command = [
            "aws",
            "ec2",
            "start-instances",
            f"--region={self.region}",
            f"--instance-ids={instance_id}",
            "--output=json",
        ]
        self.log(f"start {name} ({instance_id})")
        if self.dry_run:
            return True
        return self.run(command, timeout=None).returncode == 0

    def ssh(self, name, remote_command, *, background=False):
        if background:
            remote_command = f"nohup {remote_command} >/dev/null 2>&1 & disown"
        host = self._ssh_host(name)
        if not host:
            return 1, "", f"no address for {name}"
        command = [
            "ssh",
            # A replacement node is a new machine that may hold a recycled private
            # address, so its host key legitimately differs from last time. Pinning
            # keys here would turn every repair into a manual step; the run's own
            # secrets live on the nodes, not in this hop.
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "LogLevel=ERROR",
        ]
        if self.ssh_key:
            command += ["-i", self.ssh_key]
        command += [f"{self.ssh_user}@{host}", remote_command]
        if self.dry_run:
            self.log(f"ssh {name}: {remote_command}")
            return 0, "", ""
        result = self.run(command)
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
    def __init__(
        self, cloud, nodes, *, template, launch_command, unit, env=(),
        properties=UNIT_PROPERTIES, log=print
    ):
        self.cloud = cloud
        self.nodes = list(nodes)
        self.template = template
        self.launch_command = launch_command
        self.unit = unit
        self.properties = list(properties)
        # Extra KEY=VALUE for the unit, e.g. T2S_FABRIC=efa or WANDB_RUN_ID. These
        # have to be re-supplied on every relaunch, so they belong here and not in
        # whatever shell started the first one.
        self.env = list(env)
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
                # A GCE preemption with the default action stops the instance
                # rather than deleting it; the disk and its stale checkpoint
                # survive, which is exactly why resume reads from object storage.
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
        # A launch happens because at least one rank is down, and a torchrun job
        # missing a rank is already dead -- the survivors are blocked in a
        # collective. They still hold the unit name, and `systemd-run --unit` on a
        # loaded unit fails with "already loaded or has a fragment file", so
        # without this the first still-active node fails the whole launch and the
        # run never restarts.
        self.stop_job()
        extra = "".join(f"--setenv={shlex.quote(item)} " for item in self.env)
        limits = "".join(f"--property={shlex.quote(item)} " for item in self.properties)
        for rank, name in enumerate(self.nodes):
            remote = (
                f"sudo systemctl reset-failed {shlex.quote(self.unit)} 2>/dev/null; "
                f"sudo systemd-run --unit={shlex.quote(self.unit)} "
                f"{limits}"
                f"--setenv=T2S_MACHINE_RANK={rank} "
                f"--setenv=T2S_MAIN_IP={main_ip} "
                f"--setenv=T2S_NUM_MACHINES={len(self.nodes)} "
                f"{extra}"
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
    parser.add_argument("--cloud", choices=["gcp", "aws"], default="gcp")
    parser.add_argument("--zone", help="GCE zone; required for --cloud gcp")
    parser.add_argument("--region", help="AWS region; required for --cloud aws")
    parser.add_argument(
        "--instance-template",
        required=True,
        help=(
            "what a preempted-and-deleted node is recreated from: a GCE instance "
            "template, or an EC2 launch template name"
        ),
    )
    parser.add_argument(
        "--launch-command",
        required=True,
        help=(
            "command systemd-run executes on each node, e.g. "
            "'/opt/t2s/bin/bash scripts/train_multinode.sh --train_jsonl ...'"
        ),
    )
    parser.add_argument(
        "--setenv",
        action="append",
        default=[],
        dest="env",
        metavar="KEY=VALUE",
        help=(
            "extra environment for the unit, repeatable; re-applied on every "
            "relaunch, e.g. --setenv T2S_FABRIC=efa --setenv WANDB_RUN_ID=..."
        ),
    )
    parser.add_argument(
        "--property",
        action="append",
        default=None,
        dest="properties",
        metavar="NAME=VALUE",
        help=(
            "systemd unit property, repeatable. Default: "
            f"{' '.join(UNIT_PROPERTIES)}"
        ),
    )
    parser.add_argument("--ssh-user", default="ubuntu", help="AWS only")
    parser.add_argument("--ssh-key", help="AWS only: private key for the ssh hop")
    parser.add_argument(
        "--ssh-public-ip",
        action="store_true",
        help="AWS only: reach the nodes on their public address",
    )
    parser.add_argument("--unit", default="t2s-train")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--once", action="store_true", help="one tick, for checking a setup"
    )
    args = parser.parse_args(argv)
    for item in args.env:
        if "=" not in item:
            parser.error(f"--setenv wants KEY=VALUE, got {item!r}")

    if args.cloud == "aws":
        if not args.region:
            parser.error("--cloud aws needs --region")
        where = args.region
        cloud = Ec2Cloud(
            args.region,
            ssh_user=args.ssh_user,
            ssh_key=args.ssh_key,
            public_ip=args.ssh_public_ip,
            dry_run=args.dry_run,
        )
    else:
        if not args.zone:
            parser.error("--cloud gcp needs --zone")
        where = args.zone
        cloud = GceCloud(args.zone, dry_run=args.dry_run)
    supervisor = Supervisor(
        cloud,
        args.nodes,
        template=args.instance_template,
        launch_command=args.launch_command,
        unit=args.unit,
        env=args.env,
        properties=UNIT_PROPERTIES if args.properties is None else args.properties,
    )
    print(f"supervising {', '.join(args.nodes)} in {where}", flush=True)
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

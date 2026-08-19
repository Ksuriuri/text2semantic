import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "spot_supervisor",
    Path(__file__).resolve().parents[1] / "scripts" / "spot_supervisor.py",
)
spot_supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(spot_supervisor)

MISSING = spot_supervisor.MISSING
AMBIGUOUS = spot_supervisor.AMBIGUOUS


class FakeCloud:
    def __init__(self, states, *, ip="10.0.0.2", active=()):
        self.states = dict(states)
        self.ip = ip
        self.active = set(active)
        self.actions = []

    def status(self, name):
        return self.states.get(name, MISSING)

    def internal_ip(self, name):
        return self.ip

    def create(self, name, template):
        self.actions.append(("create", name))
        self.states[name] = "RUNNING"
        return True

    def start(self, name):
        self.actions.append(("start", name))
        self.states[name] = "RUNNING"
        return True

    def ssh(self, name, command, background=False):
        self.actions.append(("ssh", name, command))
        if command.startswith("systemctl is-active"):
            return (0, "active\n", "") if name in self.active else (3, "inactive\n", "")
        return 0, "", ""


def supervisor(cloud, nodes=("node-0", "node-1"), env=()):
    return spot_supervisor.Supervisor(
        cloud,
        nodes,
        template="t2s-a3-template",
        launch_command="bash scripts/train_multinode.sh --train_jsonl train.jsonl",
        unit="t2s-train",
        env=env,
    )


class Result:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


class FakeRun:
    """Records the commands a Cloud builds and replays canned aws output."""

    def __init__(self, instances=()):
        self.instances = list(instances)
        self.commands = []

    def __call__(self, command, *, check=False, timeout=600):
        self.commands.append(command)
        if "describe-instances" in command:
            payload = {"Reservations": [{"Instances": self.instances}]}
            return Result(json.dumps(payload))
        return Result()

    def one(self, needle):
        found = [c for c in self.commands if needle in c or needle in " ".join(c)]
        assert len(found) == 1, f"{needle}: {found}"
        return found[0]


def ec2(runner, **kwargs):
    return spot_supervisor.Ec2Cloud("us-east-2", runner=runner, **kwargs)


def test_plan_launches_when_every_node_is_up_and_idle():
    states = {"node-0": "RUNNING", "node-1": "RUNNING"}
    assert spot_supervisor.plan(states, job_running=False) == {
        "action": "launch",
        "nodes": ["node-0", "node-1"],
    }


def test_plan_watches_a_healthy_running_job():
    states = {"node-0": "RUNNING", "node-1": "RUNNING"}
    assert spot_supervisor.plan(states, job_running=True)["action"] == "watch"


def test_plan_repairs_even_while_the_job_looks_alive():
    # The surviving node's unit is still "active" for a while after its peer
    # disappears -- it is blocked in a collective waiting for ranks that are
    # gone. A live-looking job must not stop the repair.
    states = {"node-0": "RUNNING", "node-1": MISSING}
    assert spot_supervisor.plan(states, job_running=True) == {
        "action": "repair",
        "nodes": ["node-1"],
    }


def test_a_deleted_node_is_recreated_and_a_stopped_one_is_started():
    cloud = FakeCloud({"node-0": "RUNNING", "node-1": MISSING})
    assert supervisor(cloud).repair(["node-1"]) is True
    assert ("create", "node-1") in cloud.actions

    cloud = FakeCloud({"node-0": "RUNNING", "node-1": "TERMINATED"})
    assert supervisor(cloud).repair(["node-1"]) is True
    assert ("start", "node-1") in cloud.actions


def test_repair_waits_out_a_transitional_state():
    cloud = FakeCloud({"node-0": "RUNNING", "node-1": "STOPPING"})
    assert supervisor(cloud).repair(["node-1"]) is False
    assert cloud.actions == []


def test_a_tick_stops_the_job_before_repairing():
    cloud = FakeCloud(
        {"node-0": "RUNNING", "node-1": MISSING}, active=("node-0", "node-1")
    )

    assert supervisor(cloud).tick() == "repair"

    stops = [action for action in cloud.actions if "systemctl stop" in str(action)]
    creates = [action for action in cloud.actions if action[0] == "create"]
    assert len(stops) == 2
    assert cloud.actions.index(stops[0]) < cloud.actions.index(creates[0])


def test_launch_passes_rank_zero_the_main_ip_and_ranks_the_rest():
    cloud = FakeCloud({"node-0": "RUNNING", "node-1": "RUNNING"}, ip="10.140.0.9")

    assert supervisor(cloud).launch() is True

    launches = [
        command for _, _, command in
        [action for action in cloud.actions if action[0] == "ssh"]
        if "systemd-run" in command
    ]
    assert len(launches) == 2
    assert "T2S_MACHINE_RANK=0" in launches[0]
    assert "T2S_MACHINE_RANK=1" in launches[1]
    assert all("T2S_MAIN_IP=10.140.0.9" in command for command in launches)
    assert all("T2S_NUM_MACHINES=2" in command for command in launches)


def test_launch_stops_a_surviving_unit_before_starting_its_replacement():
    # One rank crashed, the others are still "active" -- blocked in a collective
    # waiting for it. systemd-run cannot reuse a unit name that is still loaded,
    # so a launch that did not stop them first would fail on the first survivor
    # and never restart the run.
    cloud = FakeCloud(
        {"node-0": "RUNNING", "node-1": "RUNNING"}, active=("node-0",),
        ip="10.140.0.9",
    )

    assert supervisor(cloud).launch() is True

    commands = [command for action, _, command in
                [a for a in cloud.actions if a[0] == "ssh"]]
    stops = [i for i, command in enumerate(commands) if "systemctl stop" in command]
    runs = [i for i, command in enumerate(commands) if "systemd-run" in command]
    assert len(stops) == 2 and len(runs) == 2
    assert max(stops) < min(runs)


def test_launch_waits_until_a_new_node_has_an_ip():
    cloud = FakeCloud({"node-0": "RUNNING", "node-1": "RUNNING"}, ip="")
    assert supervisor(cloud).launch() is False


def test_job_running_needs_every_node(monkeypatch):
    cloud = FakeCloud(
        {"node-0": "RUNNING", "node-1": "RUNNING"}, active=("node-0",)
    )
    # One node's trainer exited (a crash, or the preemption handler stopping
    # cleanly); the run is not alive just because the other is still up.
    assert supervisor(cloud).job_running() is False


RUNNING_INSTANCE = {
    "InstanceId": "i-0abc",
    "State": {"Name": "running"},
    "PrivateIpAddress": "10.1.2.3",
    "PublicIpAddress": "3.4.5.6",
}


def test_ec2_filters_by_name_tag_and_live_states():
    runner = FakeRun([RUNNING_INSTANCE])

    assert ec2(runner).status("node-0") == "RUNNING"

    command = runner.one("describe-instances")
    assert "--region=us-east-2" in command
    assert "Name=tag:Name,Values=node-0" in command
    # Server-side, because a week of Spot repairs leaves a pile of terminated
    # instances under the same name.
    assert f"Name=instance-state-name,Values={spot_supervisor._EC2_LIVE}" in command


def test_ec2_states_map_onto_the_gce_vocabulary_the_planner_reads():
    assert ec2(FakeRun([])).status("node-0") == MISSING
    stopped = dict(RUNNING_INSTANCE, State={"Name": "stopped"})
    assert ec2(FakeRun([stopped])).status("node-0") == "TERMINATED"
    # Anything transitional keeps its own name so `repair` waits instead of
    # creating a second node next to one that is still shutting down.
    pending = dict(RUNNING_INSTANCE, State={"Name": "pending"})
    assert ec2(FakeRun([pending])).status("node-0") == "PENDING"


def test_ec2_refuses_to_act_on_a_duplicated_node_name():
    twins = [RUNNING_INSTANCE, dict(RUNNING_INSTANCE, InstanceId="i-0def")]
    cloud = ec2(FakeRun(twins), log=lambda *_: None)

    assert cloud.status("node-0") == AMBIGUOUS
    # A name with two machines behind it is a bookkeeping failure, not a repair:
    # creating a third would just add $50/h.
    assert spot_supervisor.plan({"node-0": AMBIGUOUS}, job_running=False) == {
        "action": "repair",
        "nodes": ["node-0"],
    }
    assert supervisor(cloud, nodes=("node-0",)).repair(["node-0"]) is False


def test_ec2_describe_failure_is_missing_not_a_crash():
    class Failing(FakeRun):
        def __call__(self, command, **kwargs):
            self.commands.append(command)
            return Result(returncode=255, stderr="expired token")

    assert ec2(Failing(), log=lambda *_: None).status("node-0") == MISSING


def test_ec2_create_uses_the_launch_template_and_retags_the_name():
    runner = FakeRun([])

    assert ec2(runner, log=lambda *_: None).create("node-3", "t2s-p5") is True

    command = runner.one("run-instances")
    assert "LaunchTemplateName=t2s-p5" in command
    assert "--count=1" in command
    # run-instances replaces the template's tags rather than merging them, and
    # every lookup here goes through the Name tag.
    assert "ResourceType=instance,Tags=[{Key=Name,Value=node-3}]" in command


def test_ec2_start_needs_the_instance_id_not_the_name():
    runner = FakeRun([RUNNING_INSTANCE])

    assert ec2(runner, log=lambda *_: None).start("node-0") is True

    assert "--instance-ids=i-0abc" in runner.one("start-instances")


def test_ec2_rendezvous_always_uses_the_private_address():
    runner = FakeRun([RUNNING_INSTANCE])
    # Even when the supervisor itself reaches the nodes from outside the VPC: NCCL
    # bootstraps between the nodes, and a public address would leave the ranks
    # dialling out and back in.
    assert ec2(runner, public_ip=True).internal_ip("node-0") == "10.1.2.3"


def test_ec2_ssh_targets_the_address_of_the_moment():
    runner = FakeRun([RUNNING_INSTANCE])

    code, _, _ = ec2(runner, ssh_key="/keys/t2s.pem").ssh("node-0", "systemctl is-active t2s-train")

    assert code == 0
    command = runner.commands[-1]
    assert command[0] == "ssh"
    assert "-i" in command and "/keys/t2s.pem" in command
    assert "ubuntu@10.1.2.3" in command
    assert command[-1] == "systemctl is-active t2s-train"
    # A replacement node is a different machine that may hold a recycled address.
    assert "StrictHostKeyChecking=no" in command


def test_ec2_ssh_to_a_node_that_is_gone_fails_without_running_ssh():
    runner = FakeRun([])

    code, _, stderr = ec2(runner).ssh("node-0", "true")

    assert code == 1
    assert "no address" in stderr
    assert not any(command[0] == "ssh" for command in runner.commands)


def test_extra_env_is_reapplied_on_every_relaunch():
    cloud = FakeCloud({"node-0": "RUNNING", "node-1": "RUNNING"}, ip="10.1.2.3")

    assert supervisor(cloud, env=["T2S_FABRIC=efa", "WANDB_RUN_ID=t2s-v1"]).launch() is True

    launches = [c for _, _, c in
                [a for a in cloud.actions if a[0] == "ssh"] if "systemd-run" in c]
    assert len(launches) == 2
    for command in launches:
        assert "--setenv=T2S_FABRIC=efa" in command
        assert "--setenv=WANDB_RUN_ID=t2s-v1" in command


def test_the_unit_carries_the_limits_efa_needs():
    cloud = FakeCloud({"node-0": "RUNNING", "node-1": "RUNNING"}, ip="10.1.2.3")

    assert supervisor(cloud).launch() is True

    launches = [c for _, _, c in
                [a for a in cloud.actions if a[0] == "ssh"] if "systemd-run" in c]
    for command in launches:
        # Without this the transient unit gets the manager's 8 MB and EFA fails
        # its memlock gate -- on GCP the property is simply inert.
        assert "--property=LimitMEMLOCK=infinity" in command
        assert "--property=LimitNOFILE=1048576" in command


def test_main_requires_the_locator_that_matches_the_cloud():
    common = [
        "--node", "node-0",
        "--instance-template", "t2s-p5",
        "--launch-command", "bash scripts/train_multinode.sh",
        "--once",
        "--dry-run",
    ]
    for argv in (["--cloud", "aws"] + common, ["--cloud", "gcp"] + common):
        try:
            spot_supervisor.main(argv)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("a missing zone/region must not start a run")


def test_main_rejects_setenv_without_a_value():
    try:
        spot_supervisor.main([
            "--node", "node-0",
            "--zone", "us-central1-a",
            "--instance-template", "t2s-a3",
            "--launch-command", "bash scripts/train_multinode.sh",
            "--setenv", "T2S_FABRIC",
            "--once",
            "--dry-run",
        ])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("KEY without =VALUE must not reach systemd-run")

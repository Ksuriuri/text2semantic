import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "spot_supervisor",
    Path(__file__).resolve().parents[1] / "scripts" / "spot_supervisor.py",
)
spot_supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(spot_supervisor)

MISSING = spot_supervisor.MISSING


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


def supervisor(cloud, nodes=("node-0", "node-1")):
    return spot_supervisor.Supervisor(
        cloud,
        nodes,
        template="t2s-a3-template",
        launch_command="bash scripts/train_multinode.sh --train_jsonl train.jsonl",
        unit="t2s-train",
    )


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

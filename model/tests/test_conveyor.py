from collections.abc import Callable
from typing import Any, cast

import pytest
from conveyor import ConveyorParams, build, simulate
from conveyor.net import ConveyorNet
from spn import Observer, Part, Simulator, Transition


class InvariantChecker(Observer):
    def __init__(self, model: ConveyorNet) -> None:
        self.model = model
        self.intervals = 0

    def advance(self, t0: float, t1: float) -> None:
        m = self.model
        for i, b in enumerate(m.params.buffers):
            busy = m.process[i].active
            assert m.free[i].count + busy == 1
            assert m.queues[i].count + m.slots[i].count + busy == b + 1
        self.intervals += 1


class PartLog(Observer):
    def __init__(self, arrive: Transition) -> None:
        self.arrive = arrive
        self.created: list[Part] = []
        self.left = 0

    def exited(self, transition: Transition, part: Part | None, t: float) -> None:
        if transition is self.arrive and part is not None:
            self.created.append(part)

    def released(self, part: Part, t: float) -> None:
        self.left += 1


def parts_in_system(model: ConveyorNet) -> int:
    carriers = (model.to_belt, *model.accept, *model.process, *model.moves, model.back)
    queued = sum(p.count for p in (*model.belt, *model.queues))
    return queued + sum(tr.active for tr in carriers)


def test_net_matches_formal_model() -> None:
    model = build(ConveyorParams(buffers=(0, 1, 2)))
    net = model.net
    assert [p.name for p in net.places] == [
        "G", "A0", "A1", "A2", "A3", "B1", "B2", "B3", "E1", "E2", "E3",
        "F1", "F2", "F3", "N1", "N2", "N3", "R",
    ]  # fmt: skip
    assert [model.slots[i].count for i in range(3)] == [1, 2, 3]
    assert all(f.count == 1 for f in model.free)
    assert [tr.name for tr in model.moves] == ["Переїзд 1→2", "Переїзд 2→3"]
    assert all(tr.priority == 1 for tr in model.accept)
    others = [tr for tr in net.transitions if tr not in model.accept]
    assert all(tr.priority == 0 for tr in others)


@pytest.mark.parametrize("buffers", [(0, 0, 0, 0, 0), (1, 0, 2, 0, 3), (0, 0, 0, 0, 10)])
def test_invariants_hold_after_every_event(buffers: tuple[int, ...]) -> None:
    model = build(ConveyorParams(buffers=buffers))
    checker = InvariantChecker(model)
    Simulator(model.net, seed=3, observers=[checker]).run(3_000.0)
    assert checker.intervals > 10_000


@pytest.mark.parametrize("params", [ConveyorParams(), ConveyorParams.uniform(5, 1)])
def test_littles_law(params: ConveyorParams) -> None:
    r = simulate(params, warmup=1_000.0, duration=10_000.0, seed=5)
    assert r.in_system_mean == pytest.approx(r.throughput * r.sojourn_mean, rel=0.01)


def test_every_created_part_is_released_or_still_inside() -> None:
    model = build(ConveyorParams())
    log = PartLog(model.arrive)
    Simulator(model.net, seed=2, observers=[log]).run(2_000.0)
    assert len(log.created) == log.left + parts_in_system(model)
    assert sum(p.count for p in model.done) == log.left


def test_light_load_parts_served_by_first_device_without_returns() -> None:
    r = simulate(ConveyorParams(lam=0.05), warmup=1_000.0, duration=200_000.0, seed=4)
    assert r.returns_mean == 0.0
    assert r.sojourn_mean == pytest.approx(1.0, rel=0.08)
    assert r.utilization[0] == pytest.approx(0.05, rel=0.1)
    assert r.utilization[-1] < 1e-3


def test_output_matches_input_in_steady_state() -> None:
    r = simulate(ConveyorParams(), warmup=1_000.0, duration=20_000.0, seed=6)
    assert r.throughput == pytest.approx(4.0, rel=0.01)
    assert r.busy_mean == pytest.approx(4.0 * 1.0, rel=0.03)
    assert sum(r.busy_share) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("params", "low", "high"),
    [(ConveyorParams(), 8.8, 10.2), (ConveyorParams.uniform(5, 1), 3.75, 4.15)],
)
def test_sojourn_time_matches_independent_models(
    params: ConveyorParams, low: float, high: float
) -> None:
    r = simulate(params, warmup=2_000.0, duration=30_000.0, seed=8)
    assert low < r.sojourn_mean < high


def test_same_parts_in_every_buffer_configuration() -> None:
    def created(params: ConveyorParams) -> list[Part]:
        model = build(params)
        log = PartLog(model.arrive)
        Simulator(model.net, seed=11, observers=[log]).run(500.0)
        return log.created

    assert created(ConveyorParams()) == created(ConveyorParams(buffers=(0, 0, 0, 0, 10)))


def test_same_seed_reproduces_result() -> None:
    def run() -> object:
        return simulate(ConveyorParams.uniform(5, 1), warmup=100.0, duration=1_000.0, seed=9)

    assert run() == run()


def test_sojourn_samples_kept_on_request() -> None:
    r = simulate(ConveyorParams(), warmup=100.0, duration=500.0, seed=1, keep_sojourn=True)
    assert len(r.sojourn) == r.completed
    assert sum(r.sojourn) / r.completed == pytest.approx(r.sojourn_mean)


@pytest.mark.parametrize(
    "make",
    [
        lambda: ConveyorParams(buffers=()),
        lambda: ConveyorParams(buffers=(0, -1)),
        lambda: ConveyorParams(lam=0.0),
        lambda: ConveyorParams(t_ob=-1.0),
        lambda: ConveyorParams(t_ret=-5.0),
        lambda: ConveyorParams(service=cast(Any, "uniform")),
    ],
)
def test_invalid_params_rejected(make: Callable[[], ConveyorParams]) -> None:
    with pytest.raises(ValueError):
        make()


def test_overload_parts_pile_up_and_output_stays_below_input() -> None:
    def in_system(duration: float) -> float:
        params = ConveyorParams(lam=6.0)
        return simulate(params, warmup=0.0, duration=duration, seed=1).in_system_mean

    short, long = in_system(250.0), in_system(1_000.0)
    assert long > 3 * short
    r = simulate(ConveyorParams(lam=6.0), warmup=0.0, duration=1_000.0, seed=1)
    assert r.throughput < 5.0


def test_large_configuration_keeps_invariants_and_parts() -> None:
    model = build(ConveyorParams.uniform(8, 10, lam=8.0))
    checker = InvariantChecker(model)
    log = PartLog(model.arrive)
    Simulator(model.net, seed=12, observers=[checker, log]).run(2_000.0)
    assert checker.intervals > 20_000
    assert len(log.created) == log.left + parts_in_system(model)


def test_deterministic_service_light_load_gives_exact_sojourn() -> None:
    r = simulate(ConveyorParams(lam=0.05, service="det"), warmup=100.0, duration=5_000.0, seed=1)
    assert r.sojourn_mean == pytest.approx(1.0)
    assert r.sojourn_std == pytest.approx(0.0, abs=1e-9)
    assert r.returns_mean == 0

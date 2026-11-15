from collections.abc import Sequence
from random import Random

import pytest
from spn import (
    Constant,
    Exponential,
    FromPart,
    Net,
    Observer,
    Part,
    Place,
    Simulator,
    Transition,
    exponential,
)


class Recorder(Observer):
    def __init__(self) -> None:
        self.exits: list[tuple[str, float]] = []
        self.left: list[tuple[Part, float]] = []

    def exited(self, transition: Transition, part: Part | None, t: float) -> None:
        self.exits.append((transition.name, t))

    def released(self, part: Part, t: float) -> None:
        self.left.append((part, t))


def new_part(t: float, rng: Random) -> Part:
    return Part(t_in=t, tau=exponential(1.0, rng))


def test_constant_delay_cycle_exits_on_schedule() -> None:
    p = Place("P", 1)
    t = Transition("T", [p], [p], Constant(2.0))
    rec = Recorder()
    Simulator(Net([p], [t]), seed=1, observers=[rec]).run(10.0)
    assert [time for _, time in rec.exits] == [2.0, 4.0, 6.0, 8.0, 10.0]


def test_multichannel_transition_takes_all_tokens_at_once() -> None:
    p, q = Place("P", 3), Place("Q")
    t = Transition("T", [p], [q], Constant(1.0))
    sim = Simulator(Net([p, q], [t]), seed=1)
    sim.run(0.5)
    assert (p.count, t.active, q.count) == (0, 3, 0)
    sim.run(1.0)
    assert (t.active, q.count) == (0, 3)


@pytest.mark.parametrize("seed", range(20))
def test_higher_priority_wins_conflict(seed: int) -> None:
    p, q1, q2 = Place("P", 1), Place("Q1"), Place("Q2")
    high = Transition("high", [p], [q1], Constant(0.0), priority=1)
    low = Transition("low", [p], [q2], Constant(0.0))
    Simulator(Net([p, q1, q2], [low, high]), seed=seed).run(1.0)
    assert (q1.count, q2.count) == (1, 0)


def test_equal_priorities_resolved_at_random() -> None:
    wins = 0
    for seed in range(400):
        p, q1, q2 = Place("P", 1), Place("Q1"), Place("Q2")
        a = Transition("a", [p], [q1], Constant(0.0))
        b = Transition("b", [p], [q2], Constant(0.0))
        Simulator(Net([p, q1, q2], [a, b]), seed=seed).run(1.0)
        wins += q1.count
    assert 0.4 < wins / 400 < 0.6


def test_parts_leave_queue_in_arrival_order() -> None:
    parts = [Part(t_in=float(i), tau=1.0) for i in range(3)]
    queue, free = Place("Q", typed=True, parts=parts), Place("F", 1)
    serve = Transition("serve", [queue, free], [free], FromPart())
    rec = Recorder()
    Simulator(Net([queue, free], [serve]), seed=1, observers=[rec]).run(10.0)
    assert rec.left == [(parts[0], 1.0), (parts[1], 2.0), (parts[2], 3.0)]


def test_source_creates_part_when_transition_fires() -> None:
    g, a = Place("G", 1), Place("A", typed=True)
    arrive = Transition("arrive", [g], [g, a], Constant(0.5), source=new_part)
    Simulator(Net([g, a], [arrive]), seed=1).run(1.6)
    assert [part.t_in for part in a.parts] == [0.5, 1.0, 1.5]


def test_returned_part_is_taken_before_part_arriving_at_same_moment() -> None:
    returned = Part(t_in=-5.0, tau=1.0)
    away = Place("X", typed=True, parts=[returned])
    g, a0, a1, free = Place("G", 1), Place("A0", typed=True), Place("A1", typed=True), Place("F", 1)
    back = Transition("back", [away], [a1], Constant(5.0))
    arrive = Transition("arrive", [g], [g, a0], Constant(5.0), source=new_part)
    to_belt = Transition("to_belt", [a0], [a1], Constant(0.0))
    process = Transition("process", [a1, free], [free], FromPart())
    rec = Recorder()
    net = Net([away, g, a0, a1, free], [arrive, to_belt, back, process])
    Simulator(net, seed=1, observers=[rec]).run(6.5)
    assert rec.left[0] == (returned, 6.0)


def test_moments_closer_than_tolerance_are_simultaneous() -> None:
    p1, p2, p3, q1, q2 = Place("P1", 1), Place("P2"), Place("P3", 1), Place("Q1"), Place("Q2")
    first = Transition("first", [p1], [p2], Constant(0.1))
    second = Transition("second", [p2], [q1], Constant(0.2))
    direct = Transition("direct", [p3], [q2], Constant(0.3))
    rec = Recorder()
    net = Net([p1, p2, p3, q1, q2], [first, second, direct])
    Simulator(net, seed=1, observers=[rec]).run(1.0)
    times = dict(rec.exits)
    assert times["second"] == times["direct"]


class TimeAverages(Observer):
    def __init__(
        self,
        places: Sequence[Place] = (),
        transitions: Sequence[Transition] = (),
        *,
        warmup: float = 0.0,
    ) -> None:
        self._places = tuple(places)
        self._transitions = tuple(transitions)
        self._warmup = warmup
        self._place_area = [0.0] * len(self._places)
        self._transition_area = [0.0] * len(self._transitions)
        self.observed = 0.0

    def advance(self, t0: float, t1: float) -> None:
        lo = max(t0, self._warmup)
        if t1 <= lo:
            return
        dt = t1 - lo
        self.observed += dt
        for i, p in enumerate(self._places):
            self._place_area[i] += p.count * dt
        for i, tr in enumerate(self._transitions):
            self._transition_area[i] += tr.active * dt

    def place_mean(self, place: Place) -> float:
        return self._place_area[self._places.index(place)] / self.observed

    def transition_mean(self, transition: Transition) -> float:
        return self._transition_area[self._transitions.index(transition)] / self.observed


def test_mm1_queue_matches_analytic_solution() -> None:
    lam, mu = 0.5, 1.0
    rho = lam / mu
    g, queue, free = Place("G", 1), Place("Q"), Place("F", 1)
    arrive = Transition("arrive", [g], [g, queue], Exponential(1 / lam))
    serve = Transition("serve", [queue, free], [free], Exponential(1 / mu))
    stats = TimeAverages([queue, free], warmup=1_000.0)
    Simulator(Net([g, queue, free], [arrive, serve]), seed=7, observers=[stats]).run(200_000.0)
    assert 1 - stats.place_mean(free) == pytest.approx(rho, rel=0.03)
    assert stats.place_mean(queue) == pytest.approx(rho**2 / (1 - rho), rel=0.08)


def test_same_seed_reproduces_run_and_other_seed_differs() -> None:
    def run(seed: int) -> tuple[float, float]:
        g, queue, free = Place("G", 1), Place("Q"), Place("F", 1)
        arrive = Transition("arrive", [g], [g, queue], Exponential(2.0))
        serve = Transition("serve", [queue, free], [free], Exponential(1.0))
        stats = TimeAverages([queue, free])
        Simulator(Net([g, queue, free], [arrive, serve]), seed=seed, observers=[stats]).run(500.0)
        return stats.place_mean(queue), stats.place_mean(free)

    assert run(3) == run(3)
    assert run(3) != run(4)


def test_exponential_generator_has_requested_mean() -> None:
    rng = Random(1)
    n = 200_000
    assert sum(exponential(2.5, rng) for _ in range(n)) / n == pytest.approx(2.5, rel=0.01)


def test_invalid_delays_rejected() -> None:
    with pytest.raises(ValueError):
        Constant(-1.0)
    with pytest.raises(ValueError):
        Exponential(0.0)
    with pytest.raises(ValueError):
        FromPart().sample(None, Random(1))


@pytest.mark.parametrize("seed", range(10))
def test_parts_leaving_together_keep_their_order(seed: int) -> None:
    parts = [Part(t_in=float(i), tau=1.0) for i in range(4)]
    a, b = Place("A", typed=True, parts=parts), Place("B", typed=True)
    move = Transition("move", [a], [b], Constant(1.0))
    Simulator(Net([a, b], [move]), seed=seed).run(2.0)
    assert b.parts == tuple(parts)


def test_negative_marking_rejected() -> None:
    with pytest.raises(ValueError):
        Place("P", -1)
    with pytest.raises(ValueError):
        Place("P").take()


def test_negative_delay_stops_simulation() -> None:
    a, b = Place("A", typed=True, parts=[Part(t_in=0.0, tau=-1.0)]), Place("B", typed=True)
    move = Transition("move", [a], [b], FromPart())
    with pytest.raises(ValueError):
        Simulator(Net([a, b], [move]), seed=1).run(1.0)

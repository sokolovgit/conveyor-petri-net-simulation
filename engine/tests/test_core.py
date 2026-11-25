from collections.abc import Sequence
from random import Random

import pytest
from spn import Constant, Exponential, Net, Observer, Place, Simulator, Transition, exponential


class Recorder(Observer):
    def __init__(self) -> None:
        self.exits: list[tuple[str, float]] = []
        self.entries: list[tuple[str, float, float]] = []

    def entered(self, tr: Transition, t: float, moment: float) -> None:
        self.entries.append((tr.name, t, moment))

    def exited(self, tr: Transition, t: float, moment: float) -> None:
        self.exits.append((tr.name, t))


class Negative:
    stream = "delay"

    def sample(self, rng: Random) -> float:
        return -1.0


def test_constant_delay_cycle_exits_on_schedule() -> None:
    p = Place("P", 1)
    t = Transition("T", [p], [p], Constant(2.0))
    rec = Recorder()
    Simulator(Net([p], [t]), seed=1, observers=[rec]).run(10.0)
    assert [time for _, time in rec.exits] == [2.0, 4.0, 6.0, 8.0, 10.0]


def test_multichannel_transition_keeps_one_exit_moment_per_channel() -> None:
    p, q = Place("P", 3), Place("Q")
    t = Transition("T", [p], [q], Constant(1.0))
    sim = Simulator(Net([p, q], [t]), seed=1)
    sim.run(0.5)
    assert (p.count, t.exits, q.count) == (0, [1.0, 1.0, 1.0], 0)
    sim.run(1.0)
    assert (t.exits, q.count) == ([], 3)


def test_exit_moments_are_the_state_of_the_transition() -> None:
    g, a, b = Place("G", 1), Place("A"), Place("B")
    arrive = Transition("arrive", [g], [g, a], Constant(0.5))
    move = Transition("move", [a], [b], Constant(1.2))
    sim = Simulator(Net([g, a, b], [arrive, move]), seed=1)
    sim.run(1.6)
    assert move.exits == pytest.approx([1.7, 2.2, 2.7])
    assert b.count == 0


def test_arc_weights_take_and_put_given_number_of_tokens() -> None:
    p, q = Place("P", 5), Place("Q")
    t = Transition("T", [(p, 2)], [(q, 3)], Constant(1.0))
    Simulator(Net([p, q], [t]), seed=1).run(1.0)
    assert (p.count, q.count) == (1, 6)


def test_transition_needs_tokens_for_its_arc_weight() -> None:
    p, q = Place("P", 1), Place("Q")
    t = Transition("T", [(p, 2)], [q], Constant(0.0))
    Simulator(Net([p, q], [t]), seed=1).run(1.0)
    assert (p.count, q.count) == (1, 0)


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


def test_simultaneous_exits_happen_in_random_order() -> None:
    orders = set()
    for seed in range(50):
        p1, p2, q = Place("P1", 1), Place("P2", 1), Place("Q")
        a = Transition("a", [p1], [q], Constant(1.0))
        b = Transition("b", [p2], [q], Constant(1.0))
        rec = Recorder()
        Simulator(Net([p1, p2, q], [a, b]), seed=seed, observers=[rec]).run(1.0)
        orders.add(tuple(name for name, _ in rec.exits))
    assert orders == {("a", "b"), ("b", "a")}


def test_all_exits_happen_before_any_entry() -> None:
    p1, p2, q, r = Place("P1", 1), Place("P2", 1), Place("Q"), Place("R")
    a = Transition("a", [p1], [q], Constant(1.0))
    b = Transition("b", [p2], [q], Constant(1.0))
    take_two = Transition("take_two", [(q, 2)], [r], Constant(0.0))
    Simulator(Net([p1, p2, q, r], [a, b, take_two]), seed=3).run(1.0)
    assert (q.count, r.count) == (0, 1)


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


def test_named_streams_are_independent_of_other_delays() -> None:
    def service_delays(extra_delays: bool) -> list[float]:
        g, q, f, x = Place("G", 1), Place("Q"), Place("F", 1), Place("X", 1)
        arrive = Transition("arrive", [g], [g, q], Constant(1.0))
        serve = Transition("serve", [q, f], [f], Exponential(0.5, stream="service"))
        trs = [arrive, serve]
        if extra_delays:
            trs.append(Transition("noise", [x], [x], Exponential(0.3)))
        rec = Recorder()
        Simulator(Net([g, q, f, x], trs), seed=5, observers=[rec]).run(20.0)
        return [m - t for name, t, m in rec.entries if name == "serve"]

    assert service_delays(False) == service_delays(True)


class TimeAverages(Observer):
    def __init__(
        self,
        places: Sequence[Place] = (),
        *,
        warmup: float = 0.0,
    ) -> None:
        self._places = tuple(places)
        self._warmup = warmup
        self._place_area = [0.0] * len(self._places)
        self.observed = 0.0

    def advance(self, t0: float, t1: float) -> None:
        lo = max(t0, self._warmup)
        if t1 <= lo:
            return
        dt = t1 - lo
        self.observed += dt
        for i, p in enumerate(self._places):
            self._place_area[i] += p.count * dt

    def place_mean(self, place: Place) -> float:
        return self._place_area[self._places.index(place)] / self.observed


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


def test_invalid_delays_and_arcs_rejected() -> None:
    with pytest.raises(ValueError):
        Constant(-1.0)
    with pytest.raises(ValueError):
        Exponential(0.0)
    with pytest.raises(ValueError):
        Transition("T", [(Place("P"), 0)], [], Constant(1.0))


def test_net_rejects_foreign_places_and_duplicate_names() -> None:
    p, q = Place("P"), Place("Q")
    with pytest.raises(ValueError):
        Net([p], [Transition("T", [p], [q], Constant(1.0))])
    with pytest.raises(ValueError):
        Net([p, Place("P")], [])


def test_negative_marking_rejected() -> None:
    with pytest.raises(ValueError):
        Place("P", -1)
    with pytest.raises(ValueError):
        Place("P").take()


def test_negative_delay_stops_simulation() -> None:
    a, b = Place("A", 1), Place("B")
    move = Transition("move", [a], [b], Negative())
    with pytest.raises(ValueError):
        Simulator(Net([a, b], [move]), seed=1).run(1.0)

import heapq
import math
from collections.abc import Iterable, Sequence
from operator import attrgetter
from random import Random

from spn.net import Net, Place, Transition
from spn.observers import Observer

SIMULTANEITY = 1e-9

_by_index = attrgetter("index")


def _by_priority(transitions: Iterable[Transition]) -> list[list[Transition]]:
    groups: dict[int, list[Transition]] = {}
    for tr in sorted(transitions, key=_by_index):
        groups.setdefault(tr.priority, []).append(tr)
    return [groups[r] for r in sorted(groups, reverse=True)]


class Simulator:
    tolerance = SIMULTANEITY

    def __init__(self, net: Net, *, seed: int, observers: Sequence[Observer] = ()) -> None:
        self.net = net
        self.seed = seed
        self._streams: dict[str, Random] = {}
        self._order = self.stream("order")
        self._observers = tuple(observers)
        self._nearest: list[tuple[float, int, Transition]] = []
        self._seq = 0
        self._started = False
        self.t = 0.0

    def stream(self, name: str) -> Random:
        rng = self._streams.get(name)
        if rng is None:
            rng = self._streams[name] = Random(f"{self.seed}:{name}")
        return rng

    def run(self, until: float) -> None:
        if not self._started:
            self._started = True
            self._enter(self.net.transitions)
        while self.nearest() <= until + self.tolerance:
            t_next = self.nearest()
            self._collect(t_next)
            self.t = t_next
            self.transform()
        self._collect(until)
        self.t = max(self.t, until)

    def nearest(self) -> float:
        return self._nearest[0][0] if self._nearest else math.inf

    def transform(self) -> None:
        changed = self._exit()
        self._enter(self.net.consumers(changed))

    def _collect(self, t: float) -> None:
        if t > self.t:
            for obs in self._observers:
                obs.advance(self.t, t)

    def _exit(self) -> set[Place]:
        due: list[tuple[float, Transition]] = []
        while self.nearest() <= self.t + self.tolerance:
            moment, _, tr = heapq.heappop(self._nearest)
            due.append((moment, tr))
        self._order.shuffle(due)
        changed: set[Place] = set()
        for moment, tr in due:
            tr.exits.remove(moment)
            for p, w in tr.outputs:
                p.put(w)
                changed.add(p)
            for obs in self._observers:
                obs.exited(tr, self.t, moment)
        return changed

    def _enter(self, candidates: Iterable[Transition]) -> None:
        for group in _by_priority(candidates):
            while True:
                enabled = [tr for tr in group if tr.enabled()]
                if not enabled:
                    break
                if len(enabled) == 1:
                    self._fire(enabled[0])
                else:
                    self._fire(self._order.choice(enabled))

    def _fire(self, tr: Transition) -> None:
        for p, w in tr.inputs:
            p.take(w)
        delay = tr.delay.sample(self.stream(tr.delay.stream))
        if delay < 0:
            raise ValueError(f"від’ємна затримка {tr.name}: {delay}")
        moment = self.t + delay
        tr.exits.append(moment)
        self._seq += 1
        heapq.heappush(self._nearest, (moment, self._seq, tr))
        for obs in self._observers:
            obs.entered(tr, self.t, moment)

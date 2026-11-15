import heapq
from collections.abc import Iterable, Sequence
from operator import attrgetter
from random import Random
from typing import TypeAlias

from spn.net import Net, Place, Transition
from spn.observers import Observer
from spn.part import Part

SIMULTANEITY = 1e-9

_Event: TypeAlias = tuple[float, int, Transition, Part | None]

_by_index = attrgetter("index")


def _by_entry(event: _Event) -> int:
    return event[1]


def _by_priority(transitions: Iterable[Transition]) -> list[list[Transition]]:
    groups: dict[int, list[Transition]] = {}
    for tr in sorted(transitions, key=_by_index):
        groups.setdefault(tr.priority, []).append(tr)
    return [groups[r] for r in sorted(groups, reverse=True)]


class Simulator:
    def __init__(self, net: Net, *, seed: int, observers: Sequence[Observer] = ()) -> None:
        self.net = net
        self.rng = Random(seed)
        self._order = Random(f"{seed}:order")
        self._observers = tuple(observers)
        self._calendar: list[_Event] = []
        self._seq = 0
        self._started = False
        self.t = 0.0

    def run(self, until: float) -> None:
        if not self._started:
            self._started = True
            self._enter_all(self.net.transitions)
        calendar = self._calendar
        while calendar and calendar[0][0] <= until + SIMULTANEITY:
            self._advance(calendar[0][0])
            changed = self._exit_due()
            ready = {tr for p in changed for tr in p.consumers}
            self._enter_all(ready)
        self._advance(until)

    def _advance(self, t: float) -> None:
        if t > self.t:
            for obs in self._observers:
                obs.advance(self.t, t)
            self.t = t

    def _exit_due(self) -> set[Place]:
        due: list[_Event] = []
        calendar = self._calendar
        while calendar and calendar[0][0] <= self.t + SIMULTANEITY:
            due.append(heapq.heappop(calendar))
        due.sort(key=_by_entry)
        changed: set[Place] = set()
        for _, _, tr, part in due:
            self._exit(tr, part)
            changed.update(tr.outputs)
        return changed

    def _exit(self, tr: Transition, part: Part | None) -> None:
        tr.active -= 1
        if tr.source is not None:
            part = tr.source(self.t, self.rng)
        for p in tr.outputs:
            p.put(part if p.typed else None)
        for obs in self._observers:
            obs.exited(tr, part, self.t)
        if part is not None and tr.releases_part:
            for obs in self._observers:
                obs.released(part, self.t)

    def _enter_all(self, candidates: Iterable[Transition]) -> None:
        for group in _by_priority(candidates):
            while True:
                enabled = [tr for tr in group if tr.enabled()]
                if not enabled:
                    break
                if len(enabled) == 1:
                    self._enter(enabled[0])
                else:
                    self._enter(self._order.choice(enabled))

    def _enter(self, tr: Transition) -> None:
        part: Part | None = None
        for p in tr.inputs:
            taken = p.take()
            if taken is not None:
                part = taken
        delay = tr.delay.sample(part, self.rng)
        if delay < 0:
            raise ValueError(f"від'ємна затримка переходу {tr.name}: {delay}")
        self._seq += 1
        heapq.heappush(self._calendar, (self.t + delay, self._seq, tr, part))
        tr.active += 1
        for obs in self._observers:
            obs.entered(tr, part, self.t)

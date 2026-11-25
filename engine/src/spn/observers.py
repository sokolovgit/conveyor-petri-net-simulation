from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from spn.net import Place, Transition


class Observer:
    def advance(self, t0: float, t1: float) -> None:
        pass

    def entered(self, tr: Transition, t: float, moment: float) -> None:
        pass

    def exited(self, tr: Transition, t: float, moment: float) -> None:
        pass


@dataclass(frozen=True, slots=True)
class TraceRecord:
    t: float
    kind: Literal["вхід", "вихід"]
    transition: str
    moment: float
    marking: tuple[int, ...]


class Tracer(Observer):
    def __init__(self, places: Sequence[Place], *, until: float = float("inf")) -> None:
        self.places = tuple(places)
        self._until = until
        self.records: list[TraceRecord] = []

    def entered(self, tr: Transition, t: float, moment: float) -> None:
        self._log("вхід", tr, t, moment)

    def exited(self, tr: Transition, t: float, moment: float) -> None:
        self._log("вихід", tr, t, moment)

    def _log(self, kind: Literal["вхід", "вихід"], tr: Transition, t: float, moment: float) -> None:
        if t <= self._until:
            marking = tuple(p.count for p in self.places)
            self.records.append(TraceRecord(t, kind, tr.name, moment, marking))

    def lines(self) -> list[str]:
        names = [p.name for p in self.places]
        out = []
        for r in self.records:
            marking = " ".join(f"{n}={c}" for n, c in zip(names, r.marking, strict=True))
            out.append(f"{r.t:10.3f}  {r.kind:<5}  {r.transition:<20} {r.moment:10.3f}  {marking}")
        return out

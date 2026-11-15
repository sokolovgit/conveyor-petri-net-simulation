from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from spn.net import Place, Transition
from spn.part import Part


class Observer:
    def advance(self, t0: float, t1: float) -> None:
        pass

    def entered(self, transition: Transition, part: Part | None, t: float) -> None:
        pass

    def exited(self, transition: Transition, part: Part | None, t: float) -> None:
        pass

    def released(self, part: Part, t: float) -> None:
        pass


@dataclass(frozen=True, slots=True)
class TraceRecord:
    t: float
    kind: Literal["вхід", "вихід"]
    transition: str
    part: Part | None
    marking: tuple[int, ...]


class Tracer(Observer):
    def __init__(self, places: Sequence[Place], *, until: float = float("inf")) -> None:
        self.places = tuple(places)
        self._until = until
        self.records: list[TraceRecord] = []

    def entered(self, transition: Transition, part: Part | None, t: float) -> None:
        self._log("вхід", transition, part, t)

    def exited(self, transition: Transition, part: Part | None, t: float) -> None:
        self._log("вихід", transition, part, t)

    def _log(
        self, kind: Literal["вхід", "вихід"], transition: Transition, part: Part | None, t: float
    ) -> None:
        if t <= self._until:
            marking = tuple(p.count for p in self.places)
            self.records.append(TraceRecord(t, kind, transition.name, part, marking))

    def lines(self) -> list[str]:
        names = [p.name for p in self.places]
        out = []
        for r in self.records:
            who = f"деталь t_вх={r.part.t_in:g}" if r.part is not None else ""
            marking = " ".join(f"{n}={c}" for n, c in zip(names, r.marking, strict=True))
            out.append(f"{r.t:10.3f}  {r.kind:<5}  {r.transition:<20} {who:<18} {marking}")
        return out

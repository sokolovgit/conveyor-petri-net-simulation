from collections.abc import Iterable, Sequence
from typing import TypeAlias

from spn.delay import Delay


class Place:
    __slots__ = ("count", "name")

    def __init__(self, name: str, tokens: int = 0) -> None:
        if tokens < 0:
            raise ValueError(f"від’ємне маркування позиції {name}: {tokens}")
        self.name = name
        self.count = tokens

    def put(self, k: int = 1) -> None:
        self.count += k

    def take(self, k: int = 1) -> None:
        if self.count < k:
            raise ValueError(f"маркування позиції {self.name} стало б від’ємним")
        self.count -= k

    def __repr__(self) -> str:
        return f"Place({self.name!r}, {self.count})"


Arc: TypeAlias = Place | tuple[Place, int]


def _arcs(arcs: Sequence[Arc]) -> tuple[tuple[Place, int], ...]:
    out = tuple((a, 1) if isinstance(a, Place) else a for a in arcs)
    for p, w in out:
        if w < 1:
            raise ValueError(f"кратність дуги з позицією {p.name} має бути додатною: {w}")
    return out


class Transition:
    __slots__ = ("delay", "exits", "index", "inputs", "name", "outputs", "priority")

    def __init__(
        self,
        name: str,
        inputs: Sequence[Arc],
        outputs: Sequence[Arc],
        delay: Delay,
        *,
        priority: int = 0,
    ) -> None:
        self.name = name
        self.inputs = _arcs(inputs)
        self.outputs = _arcs(outputs)
        self.delay = delay
        self.priority = priority
        self.index = -1
        self.exits: list[float] = []

    def enabled(self) -> bool:
        for p, w in self.inputs:  # noqa: SIM110
            if p.count < w:
                return False
        return True

    def __repr__(self) -> str:
        return f"Transition({self.name!r})"


class Net:
    __slots__ = ("_consumers", "_places", "_transitions", "places", "transitions")

    def __init__(self, places: Sequence[Place], transitions: Sequence[Transition]) -> None:
        self.places = tuple(places)
        self.transitions = tuple(transitions)
        self._places = {p.name: p for p in self.places}
        self._transitions = {tr.name: tr for tr in self.transitions}
        if len(self._places) < len(self.places) or len(self._transitions) < len(self.transitions):
            raise ValueError("назви позицій і переходів мережі мають бути унікальними")
        known = set(self.places)
        self._consumers: dict[Place, tuple[Transition, ...]] = dict.fromkeys(self.places, ())
        for i, tr in enumerate(self.transitions):
            tr.index = i
            for p, _ in (*tr.inputs, *tr.outputs):
                if p not in known:
                    raise ValueError(f"позиція {p.name} переходу {tr.name} не належить мережі")
            for p, _ in tr.inputs:
                self._consumers[p] += (tr,)

    def consumers(self, places: Iterable[Place]) -> set[Transition]:
        return {tr for p in places for tr in self._consumers[p]}

    def place(self, name: str) -> Place:
        return self._places[name]

    def transition(self, name: str) -> Transition:
        return self._transitions[name]

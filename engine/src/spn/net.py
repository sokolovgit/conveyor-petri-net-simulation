from collections import deque
from collections.abc import Iterable, Sequence

from spn.delay import Delay
from spn.part import Part, PartSource


class Place:
    __slots__ = ("_parts", "consumers", "count", "name", "typed")

    def __init__(
        self, name: str, tokens: int = 0, *, typed: bool = False, parts: Iterable[Part] = ()
    ) -> None:
        self.name = name
        self.typed = typed
        self.consumers: tuple[Transition, ...] = ()
        self._parts: deque[Part] | None = deque(parts) if typed else None
        if tokens < 0:
            raise ValueError(f"від'ємне маркування позиції {name}: {tokens}")
        self.count = len(self._parts) if self._parts is not None else tokens

    @property
    def parts(self) -> tuple[Part, ...]:
        return tuple(self._parts or ())

    def put(self, part: Part | None) -> None:
        if self._parts is not None and part is not None:
            self._parts.append(part)
        self.count += 1

    def take(self) -> Part | None:
        if self.count == 0:
            raise ValueError(f"маркування позиції {self.name} стало б від'ємним")
        self.count -= 1
        return self._parts.popleft() if self._parts is not None else None

    def __repr__(self) -> str:
        return f"Place({self.name!r}, {self.count})"


class Transition:
    __slots__ = (
        "active",
        "delay",
        "index",
        "inputs",
        "name",
        "outputs",
        "priority",
        "releases_part",
        "source",
    )

    def __init__(
        self,
        name: str,
        inputs: Sequence[Place],
        outputs: Sequence[Place],
        delay: Delay,
        *,
        priority: int = 0,
        source: PartSource | None = None,
    ) -> None:
        self.name = name
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        self.delay = delay
        self.priority = priority
        self.source = source
        self.index = -1
        self.active = 0
        carries_part = source is not None or any(p.typed for p in self.inputs)
        self.releases_part = carries_part and not any(p.typed for p in self.outputs)

    def enabled(self) -> bool:
        for p in self.inputs:  # noqa: SIM110
            if p.count == 0:
                return False
        return True

    def __repr__(self) -> str:
        return f"Transition({self.name!r})"


class Net:
    __slots__ = ("_places", "_transitions", "places", "transitions")

    def __init__(self, places: Sequence[Place], transitions: Sequence[Transition]) -> None:
        self.places = tuple(places)
        self.transitions = tuple(transitions)
        self._places = {p.name: p for p in self.places}
        self._transitions = {tr.name: tr for tr in self.transitions}
        for i, tr in enumerate(self.transitions):
            tr.index = i
        for p in self.places:
            p.consumers = tuple(tr for tr in self.transitions if p in tr.inputs)

    def place(self, name: str) -> Place:
        return self._places[name]

    def transition(self, name: str) -> Transition:
        return self._transitions[name]

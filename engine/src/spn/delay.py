import math
from dataclasses import dataclass
from random import Random
from typing import Protocol

from spn.part import Part


def exponential(mean: float, rng: Random) -> float:
    return -mean * math.log(1.0 - rng.random())


class Delay(Protocol):
    def sample(self, part: Part | None, rng: Random) -> float: ...


@dataclass(slots=True, frozen=True)
class Constant:
    value: float

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"затримка не може бути від'ємною: {self.value}")

    def sample(self, part: Part | None, rng: Random) -> float:
        return self.value


@dataclass(slots=True, frozen=True)
class Exponential:
    mean: float

    def __post_init__(self) -> None:
        if self.mean <= 0:
            raise ValueError(f"середнє має бути додатним: {self.mean}")

    def sample(self, part: Part | None, rng: Random) -> float:
        return exponential(self.mean, rng)


@dataclass(slots=True, frozen=True)
class FromPart:
    def sample(self, part: Part | None, rng: Random) -> float:
        if part is None:
            raise ValueError("затримка FromPart потребує деталі на вході переходу")
        return part.tau

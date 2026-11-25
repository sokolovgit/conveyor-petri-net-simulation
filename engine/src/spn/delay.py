import math
from dataclasses import dataclass
from random import Random
from typing import Protocol


def exponential(mean: float, rng: Random) -> float:
    return -mean * math.log(1.0 - rng.random())


class Delay(Protocol):
    @property
    def stream(self) -> str: ...

    def sample(self, rng: Random) -> float: ...


@dataclass(slots=True, frozen=True)
class Constant:
    value: float
    stream: str = "delay"

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"затримка не може бути від’ємною: {self.value}")

    def sample(self, rng: Random) -> float:
        return self.value


@dataclass(slots=True, frozen=True)
class Exponential:
    mean: float
    stream: str = "delay"

    def __post_init__(self) -> None:
        if self.mean <= 0:
            raise ValueError(f"середнє має бути додатним: {self.mean}")

    def sample(self, rng: Random) -> float:
        return exponential(self.mean, rng)

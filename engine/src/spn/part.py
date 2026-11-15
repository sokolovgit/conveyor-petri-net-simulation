from collections.abc import Callable
from dataclasses import dataclass
from random import Random


@dataclass(slots=True, frozen=True)
class Part:
    t_in: float
    tau: float


PartSource = Callable[[float, Random], Part]

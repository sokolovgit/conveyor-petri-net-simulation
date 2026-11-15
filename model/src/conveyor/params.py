from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ConveyorParams:
    buffers: tuple[int, ...] = (0, 0, 0, 0, 0)
    lam: float = 4.0
    t_ob: float = 1.0
    t_move: float = 1.0
    t_ret: float = 5.0
    service: Literal["exp", "det"] = "exp"

    def __post_init__(self) -> None:
        if (
            not self.buffers
            or min(self.buffers) < 0
            or self.lam <= 0
            or self.t_ob <= 0
            or min(self.t_move, self.t_ret) < 0
            or self.service not in ("exp", "det")
        ):
            raise ValueError(f"некоректні параметри конвеєра: {self}")

    @property
    def n(self) -> int:
        return len(self.buffers)

    @classmethod
    def uniform(cls, n: int, buffer: int = 0, **kwargs: Any) -> "ConveyorParams":
        return cls(buffers=(buffer,) * n, **kwargs)

from itertools import pairwise

from conveyor import ConveyorParams, build
from spn import SIMULTANEITY, Observer, Simulator, Transition

UNTIL, SEED, LAMBDAS, BINARY_LAMBDA = 21_000.0, 1, (3.0, 3.5, 4.5), 4.0
WIDE, CLOSE, GAP = 1e-6, 1e-4, 1e-7


class Exits(Observer):
    def __init__(self) -> None:
        self.records: list[tuple[float, float]] = []

    def exited(self, tr: Transition, t: float, moment: float) -> None:
        self.records.append((t, moment))


def _run(lam: float, tolerance: float) -> list[tuple[float, float]]:
    engine = type("Tolerant", (Simulator,), {"tolerance": tolerance})
    exits = Exits()
    engine(build(ConveyorParams(lam=lam)), seed=SEED, observers=[exits]).run(UNTIL)
    return exits.records


def _close_pairs(records: list[tuple[float, float]]) -> int:
    times = sorted({t for t, _ in records})
    return sum(1 for a, b in pairwise(times) if b - a < CLOSE)


def study(lam: float) -> dict[str, float]:
    gaps = [abs(m - t) for t, m in _run(lam, WIDE) if m != t]
    rounding = [g for g in gaps if g < GAP]
    return {
        "lam": lam,
        "rounding_max": max(rounding),
        "rounding_count": len(rounding),
        "merged_distinct": len(gaps) - len(rounding),
        "close_exact": _close_pairs(_run(lam, 0.0)),
        "close_tolerance": _close_pairs(_run(lam, SIMULTANEITY)),
    }


def baseline() -> int:
    return _close_pairs(_run(BINARY_LAMBDA, 0.0))


if __name__ == "__main__":
    for lam in LAMBDAS:
        print(study(lam))
    print("λ = 4, без допуску:", baseline())

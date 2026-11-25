from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from spn import Observer, Simulator

from conveyor.net import build
from conveyor.params import ConveyorParams
from conveyor.stats import ConveyorResult, ConveyorStats


def simulate(
    params: ConveyorParams,
    *,
    warmup: float,
    duration: float,
    seed: int,
    keep_sojourn: bool = False,
    observers: Sequence[Observer] = (),
) -> ConveyorResult:
    model = build(params)
    stats = ConveyorStats(model, warmup=warmup, keep_sojourn=keep_sojourn)
    Simulator(model, seed=seed, observers=[stats, *observers]).run(warmup + duration)
    return stats.result()


@dataclass(frozen=True, slots=True)
class Run:
    params: ConveyorParams
    seed: int
    warmup: float
    duration: float
    keep_sojourn: bool = False


def replications(
    params: ConveyorParams,
    *,
    runs: int,
    warmup: float,
    duration: float,
    seed: int = 1,
    keep_sojourn: bool = False,
) -> list[Run]:
    return [Run(params, seed + k, warmup, duration, keep_sojourn) for k in range(runs)]


def run_all(runs: Sequence[Run], *, workers: int | None = None) -> list[ConveyorResult]:
    if workers == 1 or len(runs) == 1:
        return [_execute(r) for r in runs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_execute, runs))


def _execute(run: Run) -> ConveyorResult:
    return simulate(
        run.params,
        warmup=run.warmup,
        duration=run.duration,
        seed=run.seed,
        keep_sojourn=run.keep_sojourn,
    )

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import pandas as pd
from conveyor import ConveyorNet, ConveyorParams, Run, build, replications, run_all
from spn import Observer, Simulator

from verification.petriobj import CASES as CHECK_CASES
from verification.petriobj import REPS as CHECK_RUNS
from verification.petriobj import T as CHECK_T

WARMUP, DURATION, RUNS, SEED = 1_000.0, 20_000.0, 5, 1
LONG_DURATION = 130_000.0
LAMBDAS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
SERIES_LAMBDAS, SERIES_UNTIL, SERIES_BIN = (4.0, 6.0), 2_000.0, 25.0
EDGE_LAMBDAS, EDGE_DURATIONS = (4.85, 4.9), (20_000.0, 80_000.0)

Check = Callable[[pd.Series, pd.Series], bool]


@dataclass(frozen=True, slots=True)
class Variant:
    label: str
    params: ConveyorParams
    expected: str
    check: Check
    overloaded: bool = False


def _close(a: float, b: float, rel: float) -> bool:
    return abs(a - b) <= rel * abs(b)


def _u(s: pd.Series, i: int) -> float:
    return float(s[f"utilization_{i}"])


def _moves(s: pd.Series, params: ConveyorParams) -> float:
    return (s.sojourn_mean - params.t_ob - params.t_ret * s.returns_mean) / params.t_move


BASE = ConveyorParams()
VARIANTS = [
    Variant(
        "базові значення",
        BASE,
        "Q = λ, K = λ·t_об",
        lambda s, b: _close(s.throughput, 4.0, 0.01) and _close(s.busy_mean, 4.0, 0.03),
    ),
    Variant(
        "λ = 0,05 дет./хв",
        ConveyorParams(lam=0.05),
        "W ≈ t_об, R = 0, працює лише 1-й пристрій",
        lambda s, b: _close(s.sojourn_mean, 1.0, 0.05) and s.returns_mean == 0 and _u(s, 5) < 0.01,
    ),
    Variant(
        "λ = 2 дет./хв",
        ConveyorParams(lam=2.0),
        "U, W, R менші за базові",
        lambda s, b: _u(s, 1) < _u(b, 1) and s.sojourn_mean < b.sojourn_mean,
    ),
    Variant(
        "λ = 4,5 дет./хв",
        ConveyorParams(lam=4.5),
        "Q = λ, U, W, R більші за базові",
        lambda s, b: (
            _close(s.throughput, 4.5, 0.01)
            and _u(s, 5) > _u(b, 5)
            and s.sojourn_mean > b.sojourn_mean
            and s.returns_mean > b.returns_mean
        ),
    ),
    Variant(
        "t_об = 0,5 хв",
        ConveyorParams(t_ob=0.5),
        "K = λ·t_об = 2, W і R менші за базові",
        lambda s, b: _close(s.busy_mean, 2.0, 0.03) and s.sojourn_mean < b.sojourn_mean,
    ),
    Variant(
        "t_об = 1,05 хв",
        ConveyorParams(t_ob=1.05),
        "K = λ·t_об = 4,2, W і R більші за базові",
        lambda s, b: (
            _close(s.busy_mean, 4.2, 0.03)
            and s.sojourn_mean > b.sojourn_mean
            and s.returns_mean > b.returns_mean
        ),
    ),
    Variant(
        "обробка детермінована",
        ConveyorParams(service="det"),
        "W = 2,5, σ_W = √1,25, R = 0, U = (1, 1, 1, 1, 0)",
        lambda s, b: (
            _close(s.sojourn_mean, 2.5, 1e-3)
            and _close(s.sojourn_std, 1.25**0.5, 1e-3)
            and s.returns_mean == 0
            and all(_close(_u(s, k), 1.0, 1e-3) for k in range(1, 5))
            and _u(s, 5) < 1e-3
        ),
    ),
    Variant(
        "b_i = 1",
        ConveyorParams.uniform(5, 1),
        "W і R значно менші, U1 більше",
        lambda s, b: s.sojourn_mean < b.sojourn_mean / 2 and _u(s, 1) > _u(b, 1),
    ),
    Variant(
        "b_5 = 10",
        ConveyorParams(buffers=(0, 0, 0, 0, 10)),
        "R і W менші, U5 більше",
        lambda s, b: s.returns_mean < b.returns_mean and _u(s, 5) > _u(b, 5),
    ),
    Variant(
        "n = 4",
        ConveyorParams.uniform(4),
        "Q < λ, K = Q·t_об, сталого режиму немає",
        lambda s, b: s.throughput < 0.95 * 4.0 and _close(s.busy_mean, s.throughput, 0.02),
        overloaded=True,
    ),
    Variant(
        "n = 6",
        ConveyorParams.uniform(6),
        "R менше, з'являється U6 < U5",
        lambda s, b: s.returns_mean < b.returns_mean and 0 < _u(s, 6) < _u(s, 5),
    ),
    Variant(
        "t_пов = 0 хв",
        ConveyorParams(t_ret=0.0),
        "W менше на t_пов·R",
        lambda s, b: _close(s.sojourn_mean, b.sojourn_mean - BASE.t_ret * b.returns_mean, 0.03),
    ),
    Variant(
        "t_пов = 10 хв",
        ConveyorParams(t_ret=10.0),
        "W більше на час повернень, кількість переїздів та сама",
        lambda s, b: (
            s.sojourn_mean > b.sojourn_mean
            and _close(_moves(s, ConveyorParams(t_ret=10.0)), _moves(b, BASE), 0.03)
        ),
    ),
    Variant(
        "t_пер = 0,5 хв",
        ConveyorParams(t_move=0.5),
        "W менше, кількість переїздів та сама",
        lambda s, b: (
            s.sojourn_mean < b.sojourn_mean
            and _close(_moves(s, ConveyorParams(t_move=0.5)), _moves(b, BASE), 0.03)
        ),
    ),
    Variant(
        "t_пер = 2 хв",
        ConveyorParams(t_move=2.0),
        "W більше за базове",
        lambda s, b: s.sojourn_mean > b.sojourn_mean,
    ),
]


class InSystem(Observer):
    def __init__(self, model: ConveyorNet, bin_width: float, until: float) -> None:
        self._m = model
        self._carriers = (model.to_belt, *model.accept, *model.process, *model.moves, model.back)
        self._width = bin_width
        self.area = [0.0] * round(until / bin_width)

    def advance(self, t0: float, t1: float) -> None:
        m = self._m
        count = sum(p.count for p in (*m.belt, *m.queues)) + sum(tr.active for tr in self._carriers)
        while t0 < t1:
            k = int(t0 // self._width)
            end = min(t1, (k + 1) * self._width)
            if k < len(self.area):
                self.area[k] += count * (end - t0)
            t0 = end


def _in_system_series(task: tuple[float, int]) -> list[float]:
    lam, seed = task
    model = build(ConveyorParams(lam=lam))
    observer = InSystem(model, SERIES_BIN, SERIES_UNTIL)
    Simulator(model.net, seed=seed, observers=[observer]).run(SERIES_UNTIL)
    return [a / SERIES_BIN for a in observer.area]


def _runs_frame(groups: dict[str, list[Run]]) -> pd.DataFrame:
    runs = [run for group in groups.values() for run in group]
    results = run_all(runs)
    rows, i = [], 0
    for name, group in groups.items():
        for run in group:
            rows.append({"group": name, "seed": run.seed, **results[i].row()})
            i += 1
    return pd.DataFrame(rows)


def simulate() -> tuple[pd.DataFrame, pd.DataFrame]:
    groups: dict[str, list[Run]] = {}
    for i, v in enumerate(VARIANTS):
        groups[f"v{i}"] = replications(
            v.params, runs=RUNS, warmup=WARMUP, duration=DURATION, seed=SEED
        )
    for i, lam in enumerate(LAMBDAS):
        groups[f"l{i}"] = replications(
            ConveyorParams(lam=lam), runs=RUNS, warmup=WARMUP, duration=DURATION, seed=SEED
        )
    for i, lam in enumerate(EDGE_LAMBDAS):
        for j, duration in enumerate(EDGE_DURATIONS):
            groups[f"e{i}{j}"] = replications(
                ConveyorParams(lam=lam), runs=RUNS, warmup=WARMUP, duration=duration, seed=SEED
            )
    groups["long"] = replications(BASE, runs=1, warmup=WARMUP, duration=LONG_DURATION, seed=SEED)
    groups["o"] = replications(
        ConveyorParams(lam=SERIES_LAMBDAS[1]),
        runs=RUNS,
        warmup=0.0,
        duration=SERIES_UNTIL,
        seed=SEED,
    )
    for i, (_, _, b, lam, t_ob) in enumerate(CHECK_CASES):
        groups[f"c{i}"] = replications(
            ConveyorParams(buffers=tuple(b), lam=lam, t_ob=t_ob),
            runs=CHECK_RUNS,
            warmup=CHECK_T / 10,
            duration=CHECK_T,
            seed=SEED,
        )
    print(f"{sum(map(len, groups.values()))} прогонів…", flush=True)
    runs = _runs_frame(groups)

    tasks = [(lam, SEED + k) for lam in SERIES_LAMBDAS for k in range(RUNS)]
    with ProcessPoolExecutor() as pool:
        curves = list(pool.map(_in_system_series, tasks))
    series = pd.DataFrame(
        [{"lam": lam, "t": (j + 0.5) * SERIES_BIN, "L": value}
         for (lam, _), curve in zip(tasks, curves, strict=True)
         for j, value in enumerate(curve)]
    ).groupby(["lam", "t"], as_index=False).L.mean()  # fmt: skip
    return runs, series


def check(runs: pd.DataFrame, series: pd.DataFrame) -> tuple[int, float]:
    means = runs.groupby("group").mean(numeric_only=True)
    spread = runs.groupby("group").sojourn_mean.std()
    base, failed = means.loc["v0"], 0
    for i, v in enumerate(VARIANTS):
        s = means.loc[f"v{i}"]
        ok = v.check(s, base)
        failed += not ok
        print(
            f"{'OK  ' if ok else 'FAIL'} {i + 1} {v.label:20} W={s.sojourn_mean:6.2f}  {v.expected}"
        )
    for i, v in enumerate(VARIANTS):
        s = means.loc[f"v{i}"]
        fits = all(
            0 <= s[f"queue_mean_{k}"] <= v.params.buffers[k - 1] for k in range(1, v.params.n + 1)
        )
        failed += not fits
        print(f"{'OK  ' if fits else 'FAIL'} {i + 1} заповненість буферів у межах ємності")
    over = series[series.lam == SERIES_LAMBDAS[1]].L
    grows = over.iloc[-1] > 3 * over.iloc[len(over) // 4]
    failed += not grows
    print(f"{'OK  ' if grows else 'FAIL'} перевантаження: L росте з часом")
    steady = [f"v{i}" for i, v in enumerate(VARIANTS) if not v.overloaded]
    variant_runs = runs[runs.group.isin(steady)]
    little = (
        (variant_runs.in_system_mean - variant_runs.throughput * variant_runs.sojourn_mean).abs()
        / variant_runs.in_system_mean
    ).max()
    print(f"формула Літтла: найбільше відносне відхилення {little:.2e}")
    print(
        "розкид W між прогонами, найбільший відносний: "
        f"{(spread / means.sojourn_mean)[steady].max():.3f}"
    )
    return failed, little

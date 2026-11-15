from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd
from conveyor import (
    ConveyorNet,
    ConveyorParams,
    ConveyorResult,
    Run,
    build,
    replications,
    run_all,
)
from scipy import stats
from spn import Observer, Part, Simulator

WARMUP, DURATION, OVER_DURATION, RUNS, SEED = 1_000.0, 20_000.0, 4_000.0, 80, 1
EPS_REL, PILOT_SEEDS, PILOT_UNTIL, OVER_UNTIL, PILOT_STEP = 0.10, 5, 60_000.0, 5_000.0, 250.0
SCREEN_RUNS, SCREEN_ALPHA, FINAL_TOP, TOTAL_BUFFER, DEVICES = 10, 0.001, 10, 10, range(3, 9)
FINAL_SEED, WINDOW = SEED + SCREEN_RUNS, 500.0
SAMPLE_STEP, LAGS, BINS, MIN_COUNT, ALPHA = 1_600, (1, 100, 400, 800, 1_600, 3_200), 20, 5, 0.05
BUSY_DT, BUSY_EVERY, BUSY_LAGS, LONG_WARMUP = 0.53, 28, (1, 2, 10, 17, 28), 5_000.0

BASE = ConveyorParams()
PILOT = {
    "базові значення": BASE,
    "b_i = 1": ConveyorParams.uniform(5, 1),
    "b_5 = 10": ConveyorParams(buffers=(0, 0, 0, 0, 10)),
    "n = 6": ConveyorParams.uniform(6),
    "n = 8": ConveyorParams.uniform(8),
    "n = 3": ConveyorParams.uniform(3),
    "n = 4": ConveyorParams.uniform(4),
}
LAWS = {
    "експоненціальний": (stats.expon, {"floc": 0}),
    "гамма": (stats.gamma, {"floc": 0}),
    "логнормальний": (stats.lognorm, {"floc": 0}),
}


class Released(Observer):
    def __init__(self) -> None:
        self.t: list[float] = []
        self.w: list[float] = []

    def released(self, part: Part, t: float) -> None:
        self.t.append(t)
        self.w.append(t - part.t_in)


class BusySampler(Observer):
    def __init__(self, model: ConveyorNet, step: float, start: float) -> None:
        self.free, self.step, self.next = model.free, step, start
        self.values: list[int] = []

    def advance(self, t0: float, t1: float) -> None:
        while self.next < t1:
            self.values.append(len(self.free) - sum(f.count for f in self.free))
            self.next += self.step


def overloaded(params: ConveyorParams) -> bool:
    return params.lam * params.t_ob >= params.n


def _trajectory(task: tuple[str, ConveyorParams, int]) -> pd.DataFrame:
    name, params, seed = task
    until = OVER_UNTIL if overloaded(params) else PILOT_UNTIL
    model = build(params)
    log = Released()
    Simulator(model.net, seed=seed, observers=[log]).run(until)
    t, w = np.array(log.t), np.cumsum(log.w)
    grid = np.arange(PILOT_STEP, until + PILOT_STEP / 2, PILOT_STEP)
    k = np.searchsorted(t, grid, side="right")
    return pd.DataFrame({"config": name, "seed": seed, "t": grid, "W": w[k - 1] / k, "Q": k / grid})


def trajectories(configs: Mapping[str, ConveyorParams] = PILOT) -> pd.DataFrame:
    tasks = [
        (name, params, SEED + s) for name, params in configs.items() for s in range(PILOT_SEEDS)
    ]
    with ProcessPoolExecutor() as pool:
        return pd.concat(pool.map(_trajectory, tasks), ignore_index=True)


def duration(traj: pd.DataFrame, response: str) -> float:
    table = traj.pivot(index="t", columns="seed", values=response)
    eps = EPS_REL * table.iloc[-1].mean()
    spread = table.max(axis=1) - table.min(axis=1)
    late = spread[::-1].cummax()[::-1] < eps
    return float(late.idxmax()) if late.any() else float("nan")


def windowed(traj: pd.DataFrame, response: str) -> pd.Series:
    frames = []
    for _, g in traj.sort_values("t").groupby("seed"):
        g = g[g.t % WINDOW == 0]
        k = np.concatenate([[0.0], (g.Q * g.t).to_numpy()])
        total = np.concatenate([[0.0], (g.W * g.Q * g.t).to_numpy()])
        values = np.diff(total) / np.diff(k) if response == "W" else np.diff(k) / WINDOW
        frames.append(pd.Series(values, index=g.t.to_numpy() - WINDOW))
    return pd.concat(frames, axis=1).mean(axis=1)


def transient(means: pd.Series) -> float:
    steady = means[means.index >= means.index.to_numpy().max() / 2].mean()
    early = means[means.index < OVER_UNTIL]
    inside = ((early - steady).abs() < EPS_REL * steady)[::-1].cummin()[::-1]
    return float(inside.idxmax()) if inside.any() else float("nan")


def base_runs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    results = run_all(
        replications(
            BASE, runs=RUNS, warmup=WARMUP, duration=DURATION, seed=SEED, keep_sojourn=True
        )
    )
    frame = pd.DataFrame([r.row() for r in results])
    frame["sojourn_min"] = [min(r.sojourn) for r in results]
    frame["sojourn_max"] = [max(r.sojourn) for r in results]
    picked = slice(SAMPLE_STEP - 1, None, SAMPLE_STEP)
    sample = pd.concat(
        pd.DataFrame({"run": k, "W": np.array(r.sojourn)[picked], "D": _shift(r)[picked]})
        for k, r in enumerate(results)
    )
    acf = pd.Series(
        {lag: np.mean([_acf(np.array(r.sojourn), lag) for r in results]) for lag in LAGS}
    )
    shifts = pd.Series(np.concatenate([_shift(r) for r in results[: RUNS // 2]]))
    weights = shifts.value_counts(normalize=True).sort_index()
    return frame, sample, acf, weights


def _shift(result: ConveyorResult) -> np.ndarray:
    return np.rint(np.array(result.sojourn) - np.array(result.tau)).astype(int)


def cycle_shares(sample: np.ndarray, params: ConveyorParams = BASE, groups: int = 4) -> pd.Series:
    cycle = (params.n - 1) * params.t_move + params.t_ret
    index = np.minimum(sample // cycle, groups).astype(int)
    return pd.Series(np.bincount(index, minlength=groups + 1) / len(sample))


def _acf(w: np.ndarray, lag: int) -> float:
    d = w - w.mean()
    return float((d[:-lag] * d[lag:]).mean() / (d * d).mean())


def chi2_test(sample: np.ndarray, law: str) -> dict:
    dist, fixed = LAWS[law]
    params = dist.fit(sample, **fixed)
    counts, edges = np.histogram(sample, bins=BINS, range=(sample.min(), sample.max()))
    cdf = dist.cdf(edges, *params)
    cdf[0], cdf[-1] = 0.0, 1.0
    observed, expected = _merge(counts, np.diff(cdf) * len(sample))
    free = len(params) - len(fixed)
    df = len(observed) - 1 - free
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    critical = float(stats.chi2.ppf(1 - ALPHA, df))
    return {"law": law, "params": params, "bins": len(observed), "df": df, "chi2": chi2,
            "critical": critical, "accepted": chi2 <= critical}  # fmt: skip


def mixture_cdf(w: np.ndarray, weights: pd.Series, t_ob: float = BASE.t_ob) -> np.ndarray:
    d = weights.index.to_numpy(dtype=float)
    lag = np.clip(np.asarray(w, dtype=float)[:, None] - d[None, :], 0.0, None)
    return (1.0 - np.exp(-lag / t_ob)) @ weights.to_numpy()


def mixture_test(sample: np.ndarray, weights: pd.Series) -> dict:
    edges = np.arange(0.0, np.ceil(sample.max()) + 1.0)
    counts, _ = np.histogram(sample, bins=edges)
    cdf = mixture_cdf(edges, weights)
    cdf[-1] = 1.0
    observed, expected = _merge(counts, np.diff(cdf) * len(sample))
    df = len(observed) - 1
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    critical = float(stats.chi2.ppf(1 - ALPHA, df))
    d = weights.index.to_numpy(dtype=float)
    mean = float(d @ weights.to_numpy()) + BASE.t_ob
    var = float(((d - mean + BASE.t_ob) ** 2) @ weights.to_numpy()) + BASE.t_ob**2
    return {"law": "суміш", "mean": mean, "std": var**0.5, "bins": len(observed), "df": df,
            "chi2": chi2, "critical": critical, "accepted": chi2 <= critical,
            "size": len(sample)}  # fmt: skip


def normal_test(values: np.ndarray, bins: int = 8) -> dict:
    mean, std = float(values.mean()), float(values.std(ddof=1))
    edges = stats.norm.ppf(np.linspace(0.0, 1.0, bins + 1), mean, std)
    counts = np.histogram(values, bins=np.r_[-np.inf, edges[1:-1], np.inf])[0]
    observed, expected = _merge(counts, np.full(bins, len(values) / bins))
    df = len(observed) - 1 - 2
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    critical = float(stats.chi2.ppf(1 - ALPHA, df))
    return {"mean": mean, "std": std, "bins": len(observed), "df": df, "chi2": chi2,
            "critical": critical, "accepted": chi2 <= critical}  # fmt: skip


def _merge(observed: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs, exp = list(map(float, observed)), list(expected)
    while len(obs) > 1:
        sparse = [i for i in range(len(obs)) if obs[i] < MIN_COUNT or exp[i] < MIN_COUNT]
        if not sparse:
            break
        i = sparse[0]
        j = i + 1 if i + 1 < len(obs) else i - 1
        obs[min(i, j)] += obs.pop(max(i, j))
        exp[min(i, j)] += exp.pop(max(i, j))
    return np.array(obs), np.array(exp)


def _busy_series(task: tuple[int, float]) -> np.ndarray:
    seed, step = task
    model = build(BASE)
    sampler = BusySampler(model, step, WARMUP)
    Simulator(model.net, seed=seed, observers=[sampler]).run(WARMUP + DURATION)
    return np.array(sampler.values)


def busy_runs() -> tuple[np.ndarray, pd.Series]:
    with ProcessPoolExecutor() as pool:
        dense = list(pool.map(_busy_series, [(SEED + k, BUSY_DT) for k in range(RUNS)]))
    acf = pd.Series(
        {lag * BUSY_DT: np.mean([_acf(v.astype(float), lag) for v in dense]) for lag in BUSY_LAGS}
    )
    return np.concatenate([v[::BUSY_EVERY] for v in dense]), acf


def late_start() -> pd.DataFrame:
    runs = replications(BASE, runs=RUNS, warmup=LONG_WARMUP, duration=DURATION, seed=SEED)
    return pd.DataFrame([r.row() for r in run_all(runs)])


def binomial_test(sample: np.ndarray, n: int) -> dict:
    p = float(sample.mean()) / n
    counts = np.bincount(sample, minlength=n + 1)
    observed, expected = _merge(counts, stats.binom.pmf(np.arange(n + 1), n, p) * len(sample))
    df = len(observed) - 1 - 1
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    critical = float(stats.chi2.ppf(1 - ALPHA, df))
    return {"p": p, "size": len(sample), "bins": len(observed), "df": df, "chi2": chi2,
            "critical": critical, "accepted": chi2 <= critical}  # fmt: skip


def series(params: ConveyorParams, runs: int = RUNS, seed: int = SEED) -> pd.DataFrame:
    duration = OVER_DURATION if overloaded(params) else DURATION
    results = run_all(replications(params, runs=runs, warmup=WARMUP, duration=duration, seed=seed))
    frame = pd.DataFrame([r.row() for r in results])
    frame.insert(0, "seed", range(seed, seed + runs))
    return frame


def anova(groups: Sequence[np.ndarray]) -> dict:
    y = np.array(groups)
    q, p = y.shape
    means = y.mean(axis=1)
    d_f = p * ((means - means.mean()) ** 2).sum() / (q - 1)
    d_r = ((y - means[:, None]) ** 2).sum() / (q * (p - 1))
    critical = float(stats.f.ppf(1 - ALPHA, q - 1, q * (p - 1)))
    return {"q": q, "p": p, "d_f": d_f, "d_r": d_r, "f": d_f / d_r, "critical": critical,
            "significant": d_f / d_r > critical}  # fmt: skip


def difference(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    d = a.to_numpy() - b.to_numpy()
    return float(d.mean()), float(d.std(ddof=1) / 2)


def placements(total: int = TOTAL_BUFFER, n: int = 5) -> list[tuple[int, ...]]:
    result = []
    for cuts in combinations(range(total + n - 1), n - 1):
        bounds = (-1, *cuts, total + n - 1)
        result.append(tuple(bounds[i + 1] - bounds[i] - 1 for i in range(n)))
    return result


def screen() -> pd.DataFrame:
    tasks = [
        Run(ConveyorParams(buffers=b), SEED + k, WARMUP, DURATION)
        for b in placements()
        for k in range(SCREEN_RUNS)
    ]
    results = run_all(tasks)
    return pd.DataFrame(
        {"buffers": ",".join(map(str, run.params.buffers)), "seed": run.seed,
         "W": r.sojourn_mean, "R": r.returns_mean, "Q": r.throughput}
        for run, r in zip(tasks, results, strict=True)
    )  # fmt: skip


def shortlist(screened: pd.DataFrame) -> pd.DataFrame:
    table = screened.pivot(index="seed", columns="buffers", values="W")
    best = table.mean().idxmin()
    d = table.sub(table[best], axis=0)
    summary = pd.DataFrame(
        {"W": table.mean(), "d": d.mean(), "se": d.std(ddof=1) / np.sqrt(len(table))}
    )
    limit = stats.t.ppf(1 - SCREEN_ALPHA, len(table) - 1)
    summary["kept"] = summary.d <= limit * summary.se
    return summary.sort_values("W")


def final(kept: Sequence[str]) -> pd.DataFrame:
    frames = []
    for buffers in kept:
        params = ConveyorParams(buffers=tuple(int(b) for b in buffers.split(",")))
        frame = series(params, seed=FINAL_SEED)
        frame.insert(0, "buffers", buffers)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def devices() -> pd.DataFrame:
    frames = []
    for n in DEVICES:
        for b in (0, 1):
            frame = series(ConveyorParams.uniform(n, b))
            frame.insert(0, "b", b)
            frame.insert(0, "n", n)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def decomposition(dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, b in dev[["n", "b"]].drop_duplicates().sort_values(["n", "b"]).itertuples(index=False):
        g = dev[(dev.n == n) & (dev.b == b)]
        params = ConveyorParams.uniform(int(n), int(b))
        if overloaded(params):
            continue
        wait = sum(g[f"queue_mean_{i}"] for i in range(1, params.n + 1)) / params.lam
        back = g.returns_mean * params.t_ret
        rows.append({"n": n, "b": b, "W": g.sojourn_mean.mean(), "wait": wait.mean(),
                     "back": back.mean(),
                     "move": (g.sojourn_mean - params.t_ob - back - wait).mean()})  # fmt: skip
    return pd.DataFrame(rows)

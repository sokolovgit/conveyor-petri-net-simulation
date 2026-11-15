import math
from dataclasses import asdict, dataclass

from spn import Observer, Part, Transition

from conveyor.net import ConveyorNet


@dataclass(frozen=True, slots=True)
class ConveyorResult:
    observed: float
    completed: int
    throughput: float
    utilization: tuple[float, ...]
    busy_mean: float
    busy_std: float
    busy_share: tuple[float, ...]
    queue_mean: tuple[float, ...]
    sojourn_mean: float
    sojourn_std: float
    returns_mean: float
    in_system_mean: float
    sojourn: tuple[float, ...] = ()
    tau: tuple[float, ...] = ()

    def row(self) -> dict[str, float]:
        row: dict[str, float] = {}
        for name, value in asdict(self).items():
            if name in ("sojourn", "tau"):
                continue
            if isinstance(value, tuple):
                start = 0 if name == "busy_share" else 1
                row.update({f"{name}_{i}": v for i, v in enumerate(value, start=start)})
            else:
                row[name] = value
        return row


class ConveyorStats(Observer):
    def __init__(self, model: ConveyorNet, *, warmup: float, keep_sojourn: bool = False) -> None:
        n = model.params.n
        self._m = model
        self._warmup = warmup
        self._keep = keep_sojourn
        self._carriers = (model.to_belt, *model.accept, *model.process, *model.moves, model.back)
        self._observed = 0.0
        self._free_area = [0.0] * n
        self._queue_area = [0.0] * n
        self._busy_area = 0.0
        self._busy_sq_area = 0.0
        self._busy_time = [0.0] * (n + 1)
        self._in_system_area = 0.0
        self._completed = 0
        self._returns = 0
        self._w_sum = 0.0
        self._w_sq_sum = 0.0
        self._sojourn: list[float] = []
        self._tau: list[float] = []

    def advance(self, t0: float, t1: float) -> None:
        lo = max(t0, self._warmup)
        if t1 <= lo:
            return
        dt = t1 - lo
        m = self._m
        self._observed += dt
        busy = 0
        for i, (f, q) in enumerate(zip(m.free, m.queues, strict=True)):
            self._free_area[i] += f.count * dt
            self._queue_area[i] += q.count * dt
            busy += 1 - f.count
        self._busy_area += busy * dt
        self._busy_sq_area += busy * busy * dt
        self._busy_time[busy] += dt
        waiting = sum(p.count for p in (*m.belt, *m.queues))
        moving = sum(tr.active for tr in self._carriers)
        self._in_system_area += (waiting + moving) * dt

    def exited(self, transition: Transition, part: Part | None, t: float) -> None:
        if transition is self._m.back and t >= self._warmup:
            self._returns += 1

    def released(self, part: Part, t: float) -> None:
        if t < self._warmup:
            return
        w = t - part.t_in
        self._completed += 1
        self._w_sum += w
        self._w_sq_sum += w * w
        if self._keep:
            self._sojourn.append(w)
            self._tau.append(part.tau)

    def result(self) -> ConveyorResult:
        t, n = self._observed, self._completed
        if t <= 0:
            raise ValueError("час спостереження нульовий: прогін не вийшов за перехідний період")
        busy_mean = self._busy_area / t
        w_mean = self._w_sum / n if n else math.nan
        w_var = (self._w_sq_sum - n * w_mean * w_mean) / (n - 1) if n > 1 else math.nan
        return ConveyorResult(
            observed=t,
            completed=n,
            throughput=n / t,
            utilization=tuple(1 - a / t for a in self._free_area),
            busy_mean=busy_mean,
            busy_std=math.sqrt(max(self._busy_sq_area / t - busy_mean**2, 0.0)),
            busy_share=tuple(a / t for a in self._busy_time),
            queue_mean=tuple(a / t for a in self._queue_area),
            sojourn_mean=w_mean,
            sojourn_std=math.sqrt(max(w_var, 0.0)) if n > 1 else math.nan,
            returns_mean=self._returns / n if n else math.nan,
            in_system_mean=self._in_system_area / t,
            sojourn=tuple(self._sojourn),
            tau=tuple(self._tau),
        )

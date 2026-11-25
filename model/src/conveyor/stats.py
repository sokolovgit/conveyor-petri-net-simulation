import math
from collections import deque
from dataclasses import asdict, dataclass

from spn import Observer, Place, Transition

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
    sojourn_sample_mean: float
    sojourn_std: float
    returns_mean: float
    in_system_mean: float
    sojourn: tuple[float, ...] = ()
    before_service: tuple[float, ...] = ()

    def row(self) -> dict[str, float]:
        row: dict[str, float] = {}
        for name, value in asdict(self).items():
            if name in ("sojourn", "before_service"):
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
        self._carriers = (*model.accept, *model.process, *model.moves, model.back)
        self._waiting = (*model.belt, *model.queues)
        parts = set(self._waiting)
        self._route: dict[Transition, tuple[Place, Place | None]] = {}
        for tr in self._carriers:
            source = _part_place(tr.inputs, parts)
            if source is None:
                raise ValueError(f"перехід {tr.name} не забирає деталь")
            self._route[tr] = (source, _part_place(tr.outputs, parts))
        self._arrived: dict[Place, list[float]] = {p: [] for p in self._waiting}
        self._carried: dict[tuple[Transition, float], deque[tuple[float, float]]] = {}
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
        self._before_service: list[float] = []

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
        waiting = sum(p.count for p in self._waiting)
        moving = sum(len(tr.exits) for tr in self._carriers)
        self._in_system_area += (waiting + moving) * dt

    def entered(self, tr: Transition, t: float, moment: float) -> None:
        route = self._route.get(tr)
        if route is None:
            return
        arrived = self._arrived[route[0]]
        oldest = min(range(len(arrived)), key=arrived.__getitem__)
        t_in = arrived.pop(oldest)
        carried = self._carried.setdefault((tr, moment), deque())
        carried.append((t_in, t))

    def exited(self, tr: Transition, t: float, moment: float) -> None:
        if tr is self._m.arrive:
            self._arrived[self._m.belt[0]].append(t)
            return
        key = (tr, moment)
        carried = self._carried[key]
        t_in, t_entered = carried.popleft()
        if not carried:
            del self._carried[key]
        if tr is self._m.back and t >= self._warmup:
            self._returns += 1
        out = self._route[tr][1]
        if out is not None:
            self._arrived[out].append(t_in)
        elif t >= self._warmup:
            self._release(t, t - t_in, t_entered - t_in)

    def _release(self, t: float, w: float, before_service: float) -> None:
        self._completed += 1
        self._w_sum += w
        self._w_sq_sum += w * w
        if self._keep:
            self._sojourn.append(w)
            self._before_service.append(before_service)

    def result(self) -> ConveyorResult:
        t, n = self._observed, self._completed
        if t <= 0:
            raise ValueError("час спостереження нульовий: прогін не вийшов за перехідний період")
        busy_mean = self._busy_area / t
        w_mean = self._w_sum / n if n else math.nan
        w_var = (self._w_sq_sum - n * w_mean * w_mean) / (n - 1) if n > 1 else math.nan
        accumulated = self._in_system_area / n if n else math.nan
        return ConveyorResult(
            observed=t,
            completed=n,
            throughput=n / t,
            utilization=tuple(1 - a / t for a in self._free_area),
            busy_mean=busy_mean,
            busy_std=math.sqrt(max(self._busy_sq_area / t - busy_mean**2, 0.0)),
            busy_share=tuple(a / t for a in self._busy_time),
            queue_mean=tuple(a / t for a in self._queue_area),
            sojourn_mean=accumulated,
            sojourn_sample_mean=w_mean,
            sojourn_std=math.sqrt(max(w_var, 0.0)) if n > 1 else math.nan,
            returns_mean=self._returns / n if n else math.nan,
            in_system_mean=self._in_system_area / t,
            sojourn=tuple(self._sojourn),
            before_service=tuple(self._before_service),
        )


def _part_place(arcs: tuple[tuple[Place, int], ...], parts: set[Place]) -> Place | None:
    found = [p for p, _ in arcs if p in parts]
    if len(found) > 1:
        raise ValueError(f"перехід має переносити одну деталь: {[p.name for p in found]}")
    return found[0] if found else None

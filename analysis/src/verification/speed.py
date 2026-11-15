import time
from statistics import median

from conveyor import ConveyorParams, ConveyorStats, build
from spn import Simulator
from spn.simulator import SIMULTANEITY

UNTIL, REPEATS, SEED = 21_000.0, 3, 1


class FullScan(Simulator):
    def run(self, until: float) -> None:
        if not self._started:
            self._started = True
            self._enter_all(self.net.transitions)
        calendar = self._calendar
        while calendar and calendar[0][0] <= until + SIMULTANEITY:
            self._advance(calendar[0][0])
            self._exit_due()
            self._enter_all(self.net.transitions)
        self._advance(until)


def _timed(engine: type[Simulator], params: ConveyorParams) -> tuple[float, float]:
    times, sojourn = [], 0.0
    for _ in range(REPEATS):
        model = build(params)
        stats = ConveyorStats(model, warmup=0.0)
        start = time.perf_counter()
        engine(model.net, seed=SEED, observers=[stats]).run(UNTIL)
        times.append(time.perf_counter() - start)
        sojourn = stats.result().sojourn_mean
    return median(times), sojourn


def measure(params: ConveyorParams) -> dict[str, float]:
    full, w_full = _timed(FullScan, params)
    adapted, w_adapted = _timed(Simulator, params)
    if w_full != w_adapted:
        raise RuntimeError(f"повний перегляд змінив результат: {w_full} ≠ {w_adapted}")
    return {"full": full, "adapted": adapted, "ratio": full / adapted}


if __name__ == "__main__":
    for n in (5, 8):
        print(n, measure(ConveyorParams.uniform(n)))

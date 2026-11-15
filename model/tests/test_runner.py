import csv
from pathlib import Path

import pytest
from conveyor import ConveyorParams, replications, run_all, simulate
from conveyor.__main__ import main

WARMUP, DURATION = 50.0, 300.0


def test_parallel_results_match_serial_in_task_order() -> None:
    runs = replications(ConveyorParams(), runs=4, seed=3, warmup=WARMUP, duration=DURATION)
    assert run_all(runs, workers=2) == run_all(runs, workers=1)


def test_each_run_is_an_ordinary_simulation() -> None:
    [run] = replications(
        ConveyorParams.uniform(5, 1), runs=1, seed=7, warmup=WARMUP, duration=DURATION
    )
    assert run_all([run]) == [
        simulate(ConveyorParams.uniform(5, 1), seed=7, warmup=WARMUP, duration=DURATION)
    ]


def test_configurations_share_seeds() -> None:
    a = replications(ConveyorParams(), runs=3, seed=10, warmup=WARMUP, duration=DURATION)
    b = replications(
        ConveyorParams.uniform(5, 2), runs=3, seed=10, warmup=WARMUP, duration=DURATION
    )
    assert [r.seed for r in a] == [r.seed for r in b] == [10, 11, 12]


def test_cli_run_writes_every_replication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "r.csv"
    main(["run", "--buffers", "1,0,2", "--runs", "2", "--warmup", "10", "--duration", "100",
          "--workers", "1", "--csv", str(out)])  # fmt: skip
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert [row["seed"] for row in rows] == ["1", "2"]
    assert {"utilization_1", "utilization_3", "busy_share_0", "sojourn_mean"} <= set(rows[0])
    assert "utilization_4" not in rows[0]
    assert "sojourn_mean" in capsys.readouterr().out


def test_cli_trace_stops_at_horizon(capsys: pytest.CaptureFixture[str]) -> None:
    main(["trace", "--until", "2"])
    lines = capsys.readouterr().out.splitlines()
    assert lines
    assert all(float(line.split()[0]) <= 2 for line in lines)
    assert "Надходження" in lines[0]


def test_cli_rejects_bad_buffers() -> None:
    with pytest.raises(SystemExit):
        main(["run", "--buffers", "1,x"])

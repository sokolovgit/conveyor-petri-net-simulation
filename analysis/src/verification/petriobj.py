import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from petrinet.layout import build

from verification.des import run as des_run

PROJECT = Path(__file__).resolve().parents[3]
OUT = PROJECT / "analysis" / "results" / "check"
JAR = os.environ.get(
    "PETRIOBJ_JAR",
    str(Path.home() / "sdk/petriobjmodelpaint/petri-swing-ui-2.3.0-macos/petri-swing-ui-macos.jar"),
)

CASES = [
    ("база", 5, [0] * 5, 4.0, 1.0),
    ("буфери по 1", 5, [1] * 5, 4.0, 1.0),
    ("10 перед 5-м", 5, [0, 0, 0, 0, 10], 4.0, 1.0),
    ("n = 6", 6, [0] * 6, 4.0, 1.0),
    ("λ = 0,2", 5, [0] * 5, 0.2, 1.0),
]
T = 60000.0
REPS = 20


def petriobj(path: Path) -> list[dict]:
    res = subprocess.run(
        [
            "java",
            "-cp",
            JAR,
            str(PROJECT / "petriobj" / "src" / "PetriRun.java"),
            str(T),
            str(REPS),
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(res.stdout)[str(path)]


def metrics(run: dict, n: int, lam: float) -> dict:
    p, t = run["p"], run["t"]
    done = sum(p[f"N{i}"][1] for i in range(1, n + 1))
    in_system = sum(p[f"A{i}"][0] + p[f"B{i}"][0] for i in range(1, n + 1)) + sum(
        v[0] for k, v in t.items() if k.startswith(("Обробка", "Переїзд", "Повернення"))
    )
    q = done / T
    return {
        "Q": q,
        "W": in_system / q,
        "R": p["R"][1] / done,
        "K": n - sum(p[f"F{i}"][0] for i in range(1, n + 1)),
        "U1": 1 - p["F1"][0],
        "B": sum(p[f"B{i}"][0] for i in range(1, n + 1)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, n, b, lam, t_ob in CASES:
        path = OUT / f"{name.replace(' ', '_').replace(',', '')}.pnml"
        build(n, b, lam, t_ob).write(path)
        runs = pd.DataFrame([metrics(r, n, lam) for r in petriobj(path)])
        rows.append({"case": name, "model": "petriobj", **runs.mean().to_dict()})
        des = pd.DataFrame(
            [
                des_run(lam, t_ob, n=n, b=b, T=T + T / 10, warm=T / 10, seed=s)
                for s in range(1, REPS + 1)
            ]
        )
        rows.append({"case": name, "model": "des", **des[["Q", "W", "R", "K"]].mean().to_dict()})
        print(pd.DataFrame(rows[-2:]).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "check.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())

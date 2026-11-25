import argparse
import csv
import statistics
from collections.abc import Sequence
from pathlib import Path

from spn import Simulator, Tracer

from conveyor.net import build
from conveyor.params import ConveyorParams
from conveyor.run import replications, run_all


def _buffers(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"очікується список цілих через кому: {text}") from None


def _params(args: argparse.Namespace) -> ConveyorParams:
    return ConveyorParams(
        buffers=args.buffers, lam=args.lam, t_ob=args.t_ob, t_move=args.t_move, t_ret=args.t_ret
    )


def _run(args: argparse.Namespace) -> None:
    params = _params(args)
    runs = replications(
        params, runs=args.runs, warmup=args.warmup, duration=args.duration, seed=args.seed
    )
    rows = [
        {"seed": run.seed, **result.row()}
        for run, result in zip(runs, run_all(runs, workers=args.workers), strict=True)
    ]
    print(f"{params}\n{args.runs} прогонів по {args.duration:g} хв, зерна з {args.seed}")
    print(f"{'показник':<18} {'середнє':>10} {'s':>10}")
    for name in rows[0]:
        if name != "seed":
            values = [row[name] for row in rows]
            s = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"{name:<18} {statistics.fmean(values):>10.4f} {s:>10.4f}")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"→ {args.csv}")


def _trace(args: argparse.Namespace) -> None:
    model = build(_params(args))
    tracer = Tracer((*model.belt, *model.queues, *model.free, model.returns), until=args.until)
    Simulator(model, seed=args.seed, observers=[tracer]).run(args.until)
    print("\n".join(tracer.lines()))


def _parser() -> argparse.ArgumentParser:
    model = argparse.ArgumentParser(add_help=False)
    model.add_argument("--buffers", type=_buffers, default=(0,) * 5, help="напр. 0,0,0,0,10")
    model.add_argument("--lam", type=float, default=4.0)
    model.add_argument("--t-ob", type=float, default=1.0)
    model.add_argument("--t-move", type=float, default=1.0)
    model.add_argument("--t-ret", type=float, default=5.0)
    model.add_argument("--seed", type=int, default=1)

    parser = argparse.ArgumentParser(prog="python -m conveyor")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[model], help="серія незалежних прогонів")
    run.add_argument("--runs", type=int, default=10)
    run.add_argument("--warmup", type=float, default=2_000.0)
    run.add_argument("--duration", type=float, default=40_000.0)
    run.add_argument("--workers", type=int)
    run.add_argument("--csv", type=Path, help="результати кожного прогону")
    run.set_defaults(handler=_run)

    trace = sub.add_parser("trace", parents=[model], help="протокол подій")
    trace.add_argument("--until", type=float, default=10.0)
    trace.set_defaults(handler=_trace)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        args.handler(args)
    except ValueError as e:
        raise SystemExit(f"помилка: {e}") from None


if __name__ == "__main__":
    main()

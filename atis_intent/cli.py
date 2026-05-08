"""CLI: ``python -m atis_intent train`` / ``evaluate``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the train/evaluate subcommands."""
    parser = argparse.ArgumentParser(
        prog="atis_intent",
        description="ATIS intent classification — train and evaluate",
    )
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Run training")
    train_p.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to experiment YAML (default: config.yaml)",
    )

    eval_p = sub.add_parser("evaluate", help="Evaluate a saved run directory")
    eval_p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory produced by training (contains model.pt, bundle.pkl)",
    )
    eval_p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to experiment.yaml (default: <run-dir>/experiment.yaml)",
    )

    args = parser.parse_args(argv)

    if args.command == "train":
        from .train import main as train_main

        train_main(["--config", args.config])
    elif args.command == "evaluate":
        from .evaluate import main as evaluate_main

        eval_argv = ["--run-dir", args.run_dir]
        if args.config:
            eval_argv.extend(["--config", args.config])
        evaluate_main(eval_argv)
    else:
        parser.print_help()
        sys.exit(1)

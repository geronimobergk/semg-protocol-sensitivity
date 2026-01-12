from __future__ import annotations

import argparse

from .config import load_config
from .datasets import preprocess as preprocess_mod
from .datasets import splits as splits_mod
from .datasets.ninapro_db2 import download as download_mod
from .reporting import compare as report_mod
from . import sizing as sizing_mod
from .training import loop as training_loop
from .utils.io import read_yaml
from .utils.logging import configure_logging, get_logger

LOGGER = get_logger(__name__)


def download(cfg: dict) -> None:
    dataset_cfg = cfg.get("dataset") or {}
    source = str(dataset_cfg.get("source", dataset_cfg.get("name", ""))).lower()
    if source == "fixture":
        LOGGER.info("Skipping download for fixture dataset.")
        return
    download_mod.download_db2(cfg)


def prepare(cfg: dict) -> None:
    preprocess_mod.preprocess(cfg)


def splits(cfg: dict) -> None:
    splits_mod.generate_splits(cfg)


def traineval(cfg: dict, max_jobs: int | None = None) -> None:
    training_loop.run(cfg, max_jobs=max_jobs)


def report(cfg: dict) -> None:
    report_mod.compare(cfg)


def size(
    cfg: dict,
    split_path: str | None = None,
    warmup_steps: int = 50,
    bench_train_steps: int = 500,
    bench_val_steps: int = 100,
    device: str | None = None,
    max_k: int = 4,
    max_gpus: int = 8,
    alpha: float = 1.2,
    seeds: int | None = None,
    epochs: int | None = None,
    max_jobs: int | None = None,
    output_path: str | None = None,
) -> None:
    sizing_mod.run_size(
        cfg,
        split_path=split_path,
        warmup_steps=warmup_steps,
        train_steps=bench_train_steps,
        val_steps=bench_val_steps,
        device_override=device,
        max_k=max_k,
        max_gpus=max_gpus,
        alpha=alpha,
        seeds=seeds,
        epochs=epochs,
        max_jobs=max_jobs,
        output_path=output_path,
    )


def run_pipeline(cfg: dict, max_jobs: int | None = None) -> None:
    prepare(cfg)
    splits(cfg)
    traineval(cfg, max_jobs=max_jobs)
    report(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyml-semg",
        description="sEMG CNN experiment CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_arg(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "-c",
            "--config",
            required=True,
            help="Path to YAML config",
        )
        subparser.add_argument(
            "--profile",
            default=None,
            help="Profile name or path (e.g. smoke)",
        )
        subparser.add_argument(
            "--overrides",
            action="append",
            default=[],
            help="Override YAML files applied after profile (repeatable).",
        )

    def add_train_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--max-jobs",
            type=int,
            default=None,
            help="Max parallel jobs (stub)",
        )

    download_parser = subparsers.add_parser("download", help="Download raw dataset")
    add_config_arg(download_parser)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare raw data")
    add_config_arg(prepare_parser)

    splits_parser = subparsers.add_parser("splits", help="Generate protocol splits")
    add_config_arg(splits_parser)

    traineval_parser = subparsers.add_parser(
        "traineval", help="Train and evaluate models"
    )
    add_config_arg(traineval_parser)
    add_train_args(traineval_parser)

    report_parser = subparsers.add_parser("report", help="Generate reports")
    add_config_arg(report_parser)

    size_parser = subparsers.add_parser(
        "size", help="Benchmark + probe + size resources into sizing.json"
    )
    add_config_arg(size_parser)
    size_parser.add_argument(
        "--split",
        default=None,
        help="Path to a split.json (default: first LOSO split found)",
    )
    size_parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Warmup steps before timing",
    )
    size_parser.add_argument(
        "--bench-train-steps",
        type=int,
        default=500,
        help="Training steps for the sizing benchmark",
    )
    size_parser.add_argument(
        "--bench-val-steps",
        type=int,
        default=100,
        help="Validation steps for the sizing benchmark",
    )
    size_parser.add_argument(
        "--device",
        default=None,
        help="Device override (default: from config)",
    )
    size_parser.add_argument(
        "--max-k",
        type=int,
        default=4,
        help="Max concurrent jobs per GPU to probe",
    )
    size_parser.add_argument(
        "--max-gpus",
        type=int,
        default=8,
        help="Evaluate wall-time up to this many GPUs",
    )
    size_parser.add_argument(
        "--alpha",
        type=float,
        default=1.2,
        help="Straggler factor for wall-time estimate",
    )
    size_parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override number of seeds for sizing (default: len(train.seeds) from config)",
    )
    size_parser.add_argument(
        "--epochs",
        "--max-epochs",
        dest="epochs",
        type=int,
        default=None,
        help="Override train.max_epochs for sizing (default: from config)",
    )
    size_parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Override plan.max_jobs for effective parallelism",
    )
    size_parser.add_argument(
        "--output-path",
        default=None,
        help="Write sizing JSON (default: artifacts_root/sizing/sizing.json)",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run full pipeline (prepare -> splits -> traineval -> report)",
    )
    add_config_arg(run_parser)
    add_train_args(run_parser)

    return parser


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = [read_yaml(path) or {} for path in args.overrides]
    cfg = load_config(args.config, profile=args.profile, overrides=overrides)

    if args.command == "download":
        download(cfg)
    elif args.command == "prepare":
        prepare(cfg)
    elif args.command == "splits":
        splits(cfg)
    elif args.command == "traineval":
        traineval(cfg, max_jobs=args.max_jobs)
    elif args.command == "report":
        report(cfg)
    elif args.command == "size":
        size(
            cfg,
            split_path=args.split,
            warmup_steps=args.warmup_steps,
            bench_train_steps=args.bench_train_steps,
            bench_val_steps=args.bench_val_steps,
            device=args.device,
            max_k=args.max_k,
            max_gpus=args.max_gpus,
            alpha=args.alpha,
            seeds=args.seeds,
            epochs=args.epochs,
            max_jobs=args.max_jobs,
            output_path=args.output_path,
        )
    elif args.command == "run":
        run_pipeline(cfg, max_jobs=args.max_jobs)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

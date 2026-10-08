"""Prepare a reproducible data-scale run and launch an existing trainer."""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import yaml

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

WORKSPACE = Path(os.environ.get("FEDSCALE_WORKSPACE", "/workspace/FedScal-UMamba"))
EXPERIMENTS_DIR = WORKSPACE / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one repository experiment on a deterministic data subset."
    )
    parser.add_argument(
        "--experiment",
        default="picai/umamba_mtl",
        help="Experiment below experiments/ (default: picai/umamba_mtl).",
    )
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--output-dir", default="/outputs")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stratify-key",
        default="case_pca",
        help="Training-item field used for proportional sampling; use 'none' to disable.",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--gpus", default="0", help="Comma-separated CUDA device indices.")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cache-rate", type=float)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable online W&B logging.")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the run config and sampled datalist without starting training.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the mamba-ssm CUDA smoke test instead of training.",
    )
    return parser.parse_args()


def safe_experiment_path(experiment: str) -> Path:
    path = (EXPERIMENTS_DIR / experiment).resolve()
    if not path.is_relative_to(EXPERIMENTS_DIR.resolve()):
        raise ValueError("Experiment must be a path below /workspace/experiments")
    if not (path / "trainer.py").is_file() or not (path / "config.yaml").is_file():
        raise FileNotFoundError(f"No trainer.py and config.yaml found for {experiment!r}")
    return path


def allocate_group_counts(group_sizes: dict[str, int], target: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    quotas = {key: target * size / total for key, size in group_sizes.items()}
    counts = {key: min(group_sizes[key], math.floor(quota)) for key, quota in quotas.items()}
    remaining = target - sum(counts.values())
    order = sorted(
        group_sizes,
        key=lambda key: (quotas[key] - counts[key], group_sizes[key], key),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if counts[key] < group_sizes[key]:
                counts[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("Could not allocate the requested subset size")
    return counts


def sample_training_items(
    items: list[dict[str, Any]], fraction: float, seed: int, stratify_key: str | None
) -> list[dict[str, Any]]:
    if not 0 < fraction <= 1:
        raise ValueError("--fraction must be greater than 0 and at most 1")
    if not items:
        raise ValueError("The source datalist has an empty training split")

    target = max(1, min(len(items), round(len(items) * fraction)))
    rng = random.Random(seed)

    if stratify_key and all(stratify_key in item for item in items):
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            groups.setdefault(str(item[stratify_key]), []).append(item)
        counts = allocate_group_counts(
            {key: len(group) for key, group in groups.items()}, target
        )
        selected = []
        for key in sorted(groups):
            group = list(groups[key])
            rng.shuffle(group)
            selected.extend(group[: counts[key]])
    else:
        selected = list(items)
        rng.shuffle(selected)
        selected = selected[:target]

    rng.shuffle(selected)
    return selected


def load_source_config(experiment_dir: Path) -> dict[str, Any]:
    with (experiment_dir / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must contain a YAML mapping")
    return config


def default_run_name(experiment: str, fraction: float, seed: int) -> str:
    fraction_label = f"{fraction:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{experiment.replace('/', '-')}-f{fraction_label}-s{seed}"


def sanitize_run_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-.")
    if not sanitized:
        raise ValueError("--run-name must contain at least one letter or number")
    return sanitized


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        os.execv(sys.executable, [sys.executable, "/opt/training/mamba_smoke_test.py"])

    experiment_dir = safe_experiment_path(args.experiment)
    config = load_source_config(experiment_dir)
    source_datalist = (experiment_dir / config["data"]["json_list"]).resolve()
    if not source_datalist.is_file():
        raise FileNotFoundError(f"Source datalist not found: {source_datalist}")

    with source_datalist.open() as handle:
        datalist = json.load(handle)
    original_training = list(datalist.get("training", []))
    stratify_key = None if args.stratify_key.lower() == "none" else args.stratify_key
    datalist["training"] = sample_training_items(
        original_training, args.fraction, args.seed, stratify_key
    )

    run_name = sanitize_run_name(
        args.run_name or default_run_name(args.experiment, args.fraction, args.seed)
    )
    output_root = Path(args.output_dir).resolve()
    run_dir = output_root / run_name
    runtime_dir = run_dir / "runtime"
    checkpoints_dir = run_dir / "checkpoints"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    sampled_datalist = run_dir / "datalist.json"
    with sampled_datalist.open("w") as handle:
        json.dump(datalist, handle, indent=2)

    config["seed"] = args.seed
    config["gpus"] = [int(device.strip()) for device in args.gpus.split(",")]
    config["num_workers"] = args.num_workers
    config["data"]["data_dir"] = str(Path(args.data_dir).resolve())
    config["data"]["json_list"] = str(sampled_datalist)
    config["logger"]["active"] = args.wandb
    config["logger"]["experiment_name"] = run_name
    config["logger"]["resume_wandb_id"] = None
    if args.wandb_project:
        config["logger"]["project"] = args.wandb_project
    if args.wandb_entity:
        config["logger"]["entity"] = args.wandb_entity
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.cache_rate is not None:
        config["cache_rate"] = args.cache_rate
    if args.fast_dev_run:
        config["fast_dev_run"] = True
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    config["resume_ckpt"] = bool(args.resume)

    runtime_config = runtime_dir / "config.yaml"
    with runtime_config.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    trained_models_link = runtime_dir / "trained_models"
    if trained_models_link.is_symlink() and trained_models_link.resolve() != checkpoints_dir:
        trained_models_link.unlink()
    if not trained_models_link.exists():
        trained_models_link.symlink_to(checkpoints_dir, target_is_directory=True)

    summary = {
        "experiment": args.experiment,
        "source_datalist": str(source_datalist),
        "fraction": args.fraction,
        "seed": args.seed,
        "stratify_key": stratify_key,
        "original_training_samples": len(original_training),
        "selected_training_samples": len(datalist["training"]),
        "validation_samples": len(datalist.get("validation", [])),
        "test_samples": len(datalist.get("test", [])),
        "runtime_config": str(runtime_config),
    }
    with (run_dir / "run_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    if args.prepare_only:
        print("Prepared run files; training was not started (--prepare-only).")
        return

    os.chdir(runtime_dir)
    os.execv(sys.executable, [sys.executable, str(experiment_dir / "trainer.py")])


if __name__ == "__main__":
    main()

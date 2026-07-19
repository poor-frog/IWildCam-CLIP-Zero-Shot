from __future__ import annotations

import argparse
from pathlib import Path

from src.models.lesvi_candidate_preflight import run_candidate_preflight


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen LESVI-v0 confirmation candidates using metadata only.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    selection = run_candidate_preflight(args.registry, args.output_dir)
    print(
        "LESVI candidate preflight "
        f"{selection['status']}: selected={selection['selected_dataset_id']!r}, "
        f"reason={selection['reason']!r}"
    )


if __name__ == "__main__":
    main(parse_arguments())

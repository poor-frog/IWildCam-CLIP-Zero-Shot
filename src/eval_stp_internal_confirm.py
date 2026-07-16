import argparse
from pathlib import Path

from src.models.stp_confirmation import verify_confirmation_ready


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validate a frozen STP internal-confirmation specification without evaluating data.")
    parser.add_argument("--frozen-spec", type=Path, required=True)
    parser.add_argument("--confirmation-ledger", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-only", action="store_true", required=True)
    return parser.parse_args()


def main(args):
    ready = verify_confirmation_ready(args.frozen_spec, args.confirmation_ledger, args.workspace_root)
    print(f"Confirmation guard passed for frozen specification {ready.spec_sha256}.")


if __name__ == "__main__":
    main(parse_arguments())

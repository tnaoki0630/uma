import argparse
import json
from pathlib import Path

from uma_api.g1_data import build_g1_demo_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compact static G1 validation catalog")
    parser.add_argument("--source", type=Path, required=True, help="Extracted G1 dataset directory")
    parser.add_argument(
        "--output", type=Path, default=Path("web/g1-demo-catalog.json")
    )
    parser.add_argument("--validation-year", type=int, default=2025)
    parser.add_argument("--max-races", type=int)
    parser.add_argument("--samples", type=int, default=15_000)
    parser.add_argument("--history-limit", type=int, default=40)
    args = parser.parse_args()

    catalog = build_g1_demo_catalog(
        args.source,
        validation_year=args.validation_year,
        max_races=args.max_races,
        samples=args.samples,
        history_limit=args.history_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Translate end-effector and base positions in the robot training CSV.

Rules
- end_pos_x/y/z are shifted by (dx, dy, dz).
- The paired base coordinates move in the opposite direction with axis mapping:
  * end_pos_x ↔ -base_pos_y  → base_pos_y -= dx
  * end_pos_y ↔ -base_pos_x  → base_pos_x -= dy
  * end_pos_z ↔ -base_pos_z  → base_pos_z -= dz
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REQUIRED_COLS = (
    "end_pos_x",
    "end_pos_y",
    "end_pos_z",
    "base_pos_x",
    "base_pos_y",
    "base_pos_z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate end/base positions in a CSV with paired sign constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/robot_training_data.csv"),
        help="Path to the source CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/robot_training_data_translated.csv"),
        help="Path to write the translated CSV",
    )
    parser.add_argument("--dx", type=float, required=True, help="Translation along end_pos_x")
    parser.add_argument("--dy", type=float, required=True, help="Translation along end_pos_y")
    parser.add_argument("--dz", type=float, required=True, help="Translation along end_pos_z")
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")


def apply_translation(df: pd.DataFrame, dx: float, dy: float, dz: float) -> pd.DataFrame:
    validate_columns(df)

    df = df.copy()

    # Shift end-effector positions.
    df["end_pos_x"] = df["end_pos_x"] + dx
    df["end_pos_y"] = df["end_pos_y"] + dy
    df["end_pos_z"] = df["end_pos_z"] + dz

    # Apply paired opposite shifts to base positions with axis swap as noted above.
    df["base_pos_y"] = df["base_pos_y"] - dx
    df["base_pos_x"] = df["base_pos_x"] - dy
    df["base_pos_z"] = df["base_pos_z"] - dz

    return df


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    translated = apply_translation(df, args.dx, args.dy, args.dz)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    translated.to_csv(args.output, index=False)

    print(
        "Translation applied: dx={dx}, dy={dy}, dz={dz}. Rows written: {rows}. Output: {out}".format(
            dx=args.dx, dy=args.dy, dz=args.dz, rows=len(translated), out=args.output
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

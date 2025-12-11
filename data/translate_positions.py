#!/usr/bin/env python3
"""Generate augmented CSV data by randomly translating positions.

Rules
- end_pos_x/y/z are shifted by (dx, dy, dz).
- The paired base coordinates move oppositely with axis mapping:
  * end_pos_x ↔ -base_pos_y  → base_pos_y -= dx
  * end_pos_y ↔ -base_pos_x  → base_pos_x -= dy
  * end_pos_z ↔ -base_pos_z  → base_pos_z -= dz
- Offsets are random integers in [-offset_range, offset_range].
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
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
        description="Randomly translate end/base positions and write an augmented CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("robot_training_data.csv"),
        help="Source CSV path (relative to this file by default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("robot_training_data_random_shifted.csv"),
        help="Destination CSV path",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100_000,
        help="Number of augmented rows to generate (random mode only)",
    )
    parser.add_argument(
        "--offset-range",
        type=int,
        default=20,
        help="Max absolute integer offset applied to x/y/z",
    )
    parser.add_argument(
        "--mode",
        choices=["random", "exhaustive"],
        default="random",
        help="random: sample offsets uniformly; exhaustive: enumerate all integer offsets in range",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")


def generate_offsets(n: int, offset_range: int, seed: int | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    offsets = rng.integers(-offset_range, offset_range + 1, size=(n, 3))
    return offsets[:, 0], offsets[:, 1], offsets[:, 2]


def generate_exhaustive_offsets(offset_range: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Produce all integer triples in [-r, r]^3 without repetition.
    values = np.arange(-offset_range, offset_range + 1, dtype=int)
    grid = np.stack(np.meshgrid(values, values, values, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid[:, 0], grid[:, 1], grid[:, 2]


def apply_translation(df: pd.DataFrame, dx, dy, dz) -> pd.DataFrame:
    validate_columns(df)
    df = df.copy()

    # Coerce to Series for index alignment when arrays are used.
    dx_s = pd.Series(dx, index=df.index)
    dy_s = pd.Series(dy, index=df.index)
    dz_s = pd.Series(dz, index=df.index)

    df["end_pos_x"] = df["end_pos_x"] + dx_s
    df["end_pos_y"] = df["end_pos_y"] + dy_s
    df["end_pos_z"] = df["end_pos_z"] + dz_s

    df["base_pos_y"] = df["base_pos_y"] - dx_s
    df["base_pos_x"] = df["base_pos_x"] - dy_s
    df["base_pos_z"] = df["base_pos_z"] - dz_s

    return df


def main() -> None:
    args = parse_args()

    input_path = args.input if args.input.is_absolute() else Path(__file__).parent / args.input
    output_path = args.output if args.output.is_absolute() else Path(__file__).parent / args.output

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df)

    if args.mode == "random":
        # Sample with replacement to reach requested count.
        sampled = df.sample(n=args.count, replace=True, random_state=args.seed).reset_index(drop=True)
        dx, dy, dz = generate_offsets(args.count, args.offset_range, args.seed)
        translated = apply_translation(sampled, dx, dy, dz)
    else:  # exhaustive
        dx, dy, dz = generate_exhaustive_offsets(args.offset_range)
        num_offsets = len(dx)
        # Repeat base data for every offset; ensures no offset duplicates.
        sampled = df.loc[df.index.repeat(num_offsets)].reset_index(drop=True)
        # Tile offsets to align with repeated rows.
        dx_full = np.tile(dx, len(df))
        dy_full = np.tile(dy, len(df))
        dz_full = np.tile(dz, len(df))
        translated = apply_translation(sampled, dx_full, dy_full, dz_full)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    translated.to_csv(output_path, index=False)

    print(
        "Generated {count} rows with {mode} integer offsets in ±{rng}. Output: {out}".format(
            count=len(translated), mode=args.mode, rng=args.offset_range, out=output_path
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

#!/usr/bin/env python3
"""Augment full group sequences by translating end/base positions.

Similar to translate_positions.py but operates on whole sequences (group_id).
Each augmentation:
- Picks a source group sequence.
- Applies a single (dx, dy, dz) translation to all its rows.
- Assigns a new group_id and re-numbers point_index from 0..L-1.
- Writes augmented rows (optionally together with the originals) to a new CSV.

Translation rule (same as translate_positions):
  end_pos_x += dx; end_pos_y += dy; end_pos_z += dz
  base_pos_y -= dx; base_pos_x -= dy; base_pos_z -= dz
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLS = (
    "group_id",
    "point_index",
    "end_pos_x",
    "end_pos_y",
    "end_pos_z",
    "base_pos_x",
    "base_pos_y",
    "base_pos_z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment sequences (group_id) by translating whole trajectories.",
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
        default=Path("robot_training_data_seq_augmented.csv"),
        help="Destination CSV path",
    )
    parser.add_argument(
        "--mode",
        choices=["random", "exhaustive"],
        default="random",
        help="random: sample offsets uniformly; exhaustive: enumerate all integer offsets in range",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of augmented sequences to generate (random mode only)",
    )
    parser.add_argument(
        "--offset-range",
        type=int,
        default=20,
        help="Max absolute integer offset applied to x/y/z",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility (random mode)",
    )
    parser.add_argument(
        "--start-group-id",
        type=int,
        default=None,
        help="Override starting group_id for augmented sequences; default = max(group_id)+1",
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="Include original data rows in the output along with augmented rows",
    )
    parser.add_argument(
        "--apply-all-groups",
        action="store_true",
        help="Apply every generated offset to every group (cartesian product). Default: on for exhaustive, off for random",
    )
    parser.add_argument(
        "--keep-zero-offset",
        action="store_true",
        help="Include zero offset (0,0,0) in augmentation. Default: drop it to avoid duplicating original rows",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")


def generate_offsets_random(n: int, offset_range: int, seed: int | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    offsets = rng.integers(-offset_range, offset_range + 1, size=(n, 3))
    return offsets[:, 0], offsets[:, 1], offsets[:, 2]


def generate_offsets_exhaustive(offset_range: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = np.arange(-offset_range, offset_range + 1, dtype=int)
    grid = np.stack(np.meshgrid(vals, vals, vals, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid[:, 0], grid[:, 1], grid[:, 2]


def filter_zero_offset(dxs: np.ndarray, dys: np.ndarray, dzs: np.ndarray, keep_zero: bool):
    if keep_zero:
        return dxs, dys, dzs
    mask = ~((dxs == 0) & (dys == 0) & (dzs == 0))
    dxs, dys, dzs = dxs[mask], dys[mask], dzs[mask]
    if len(dxs) == 0:
        raise SystemExit("All offsets filtered out (zero only). Use --keep-zero-offset to retain it.")
    return dxs, dys, dzs


def translate_sequence(g: pd.DataFrame, dx, dy, dz) -> pd.DataFrame:
    g = g.copy()
    g["end_pos_x"] = g["end_pos_x"] + dx
    g["end_pos_y"] = g["end_pos_y"] + dy
    g["end_pos_z"] = g["end_pos_z"] + dz

    g["base_pos_y"] = g["base_pos_y"] - dx
    g["base_pos_x"] = g["base_pos_x"] - dy
    g["base_pos_z"] = g["base_pos_z"] - dz
    return g


def augment_sequences(
    df: pd.DataFrame,
    offsets: Tuple[np.ndarray, np.ndarray, np.ndarray],
    start_gid: int,
    start_point_idx: int,
    apply_all_groups: bool,
) -> pd.DataFrame:
    groups: Dict[int, pd.DataFrame] = {gid: g.reset_index(drop=True) for gid, g in df.groupby("group_id")}
    if not groups:
        raise SystemExit("No groups found in input")

    dxs, dys, dzs = offsets
    augmented: List[pd.DataFrame] = []
    gid_next = start_gid
    pid_next = start_point_idx

    gids = list(groups.keys())
    if apply_all_groups:
        for src_gid in gids:
            g = groups[src_gid]
            for dx, dy, dz in zip(dxs, dys, dzs):
                aug = translate_sequence(g, dx, dy, dz).copy()
                aug["group_id"] = gid_next
                aug["point_index"] = np.arange(pid_next, pid_next + len(aug), dtype=int)
                augmented.append(aug)
                gid_next += 1
                pid_next += len(aug)
    else:
        n_groups = len(gids)
        for i in range(len(dxs)):
            src_gid = gids[i % n_groups]
            g = groups[src_gid]
            aug = translate_sequence(g, dxs[i], dys[i], dzs[i]).copy()
            aug["group_id"] = gid_next
            aug["point_index"] = np.arange(pid_next, pid_next + len(aug), dtype=int)
            augmented.append(aug)
            gid_next += 1
            pid_next += len(aug)

    return pd.concat(augmented, axis=0, ignore_index=True)


def main() -> None:
    args = parse_args()

    input_path = args.input if args.input.is_absolute() else Path(__file__).parent / args.input
    output_path = args.output if args.output.is_absolute() else Path(__file__).parent / args.output

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df)

    # Resolve start group id
    start_gid = args.start_group_id
    if start_gid is None:
        if df["group_id"].dtype.kind in {"i", "u"}:
            start_gid = int(df["group_id"].max()) + 1
        else:
            raise SystemExit("group_id must be numeric to auto-assign start_group_id")

    start_point_idx = int(df["point_index"].max()) + 1

    # Build offsets
    if args.mode == "random":
        dxs, dys, dzs = generate_offsets_random(args.count, args.offset_range, args.seed)
    else:
        dxs, dys, dzs = generate_offsets_exhaustive(args.offset_range)
    dxs, dys, dzs = filter_zero_offset(dxs, dys, dzs, keep_zero=args.keep_zero_offset)
    offsets = (dxs, dys, dzs)

    apply_all = args.apply_all_groups or (args.mode == "exhaustive")

    augmented = augment_sequences(df, offsets, start_gid, start_point_idx, apply_all)

    if args.include_original:
        out_df = pd.concat([df, augmented], axis=0, ignore_index=True)
    else:
        out_df = augmented

    out_df.to_csv(output_path, index=False)
    print(
        f"Augmented sequences written: {len(augmented)} rows from {len(dxs)} new groups. "
        f"Output: {output_path} | start_group_id={start_gid} | include_original={args.include_original}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)

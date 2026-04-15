#!/usr/bin/env python3
"""Print per-depth node counts for a sklearn DecisionTree saved as joblib."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import joblib
from sklearn.tree import BaseDecisionTree


def node_depths_sklearn(tree) -> Counter[int]:
    """Return Counter mapping depth -> number of nodes (internal + leaves)."""
    children_left = tree.children_left
    children_right = tree.children_right
    n_nodes = tree.node_count

    depths: Counter[int] = Counter()
    stack: list[tuple[int, int]] = [(0, 0)]  # (node_id, depth)

    while stack:
        node_id, depth = stack.pop()
        depths[depth] += 1

        left = children_left[node_id]
        right = children_right[node_id]
        if left == -1:  # leaf
            continue
        stack.append((right, depth + 1))
        stack.append((left, depth + 1))

    # Sanity: should visit every node exactly once
    if sum(depths.values()) != n_nodes:
        raise RuntimeError(
            f"Traversal visited {sum(depths.values())} nodes, expected {n_nodes}"
        )

    return depths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count sklearn decision tree nodes per depth."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default=Path(__file__).resolve().parent / "bnn_surrogate_tree.joblib",
        type=Path,
        help="Path to .joblib file containing a DecisionTreeClassifier/Regressor",
    )
    parser.add_argument(
        "--csv",
        "-o",
        type=Path,
        metavar="PATH",
        default=None,
        help="Write the nodes-per-depth table as CSV (columns: depth, nodes).",
    )
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    if not isinstance(model, BaseDecisionTree):
        print(
            f"Expected sklearn BaseDecisionTree, got {type(model).__name__}",
            file=sys.stderr,
        )
        return 1

    depths = node_depths_sklearn(model.tree_)

    max_d = max(depths)
    total = sum(depths.values())

    print(f"Model: {type(model).__name__}")
    print(f"File:  {args.model_path.resolve()}")
    print(f"Total nodes: {total} (internal + leaves)")
    print(f"Max depth (deepest leaf): {model.get_depth()}")
    print()
    print("Nodes per depth (depth 0 = root):")
    for d in range(max_d + 1):
        print(f"  depth {d:4d}: {depths[d]:6d}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["depth", "nodes"])
            for d in range(max_d + 1):
                w.writerow([d, depths[d]])
        print()
        print(f"Wrote CSV: {args.csv.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

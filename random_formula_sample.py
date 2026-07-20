from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


def collect_expressions(extractions_dir: Path) -> list[dict[str, object]]:
	expressions: list[dict[str, object]] = []

	for json_path in sorted(extractions_dir.glob("page_*.json")):
		with json_path.open("r", encoding="utf-8") as handle:
			page_data = json.load(handle)

		page_number = page_data.get("page_number")
		for block in page_data.get("content_blocks", []):
			block_index = block.get("block_index")
			for expression in block.get("math_expressions", []):
				expressions.append(
					{
						"page_number": page_number,
						"block_index": block_index,
						"math_expression": expression,
					}
				)

	return expressions


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Sample random math expressions from cached page extractions."
	)
	parser.add_argument(
		"--count",
		type=int,
		default=50,
		help="Number of random expressions to output (default: 50).",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Optional random seed for reproducible sampling.",
	)
	parser.add_argument(
		"--extractions-dir",
		type=Path,
		default=Path("cache/extractions"),
		help="Directory containing page_*.json extraction files.",
	)
	args = parser.parse_args()

	expressions = collect_expressions(args.extractions_dir)
	if not expressions:
		raise SystemExit(f"No math expressions found in {args.extractions_dir}")

	if args.seed is not None:
		random.seed(args.seed)

	sample_size = min(args.count, len(expressions))
	sample = random.sample(expressions, sample_size)

	writer = csv.writer(sys.stdout)
	writer.writerow(["page_number", "block_index", "math_expression"])
	for expression in sample:
		writer.writerow(
			[
				expression["page_number"],
				expression["block_index"],
				expression["math_expression"],
			]
		)


if __name__ == "__main__":
	main()
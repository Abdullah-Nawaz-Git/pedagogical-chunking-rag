from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


def collect_diagrams(extractions_dir: Path) -> list[dict[str, object]]:
	diagrams: list[dict[str, object]] = []

	for json_path in sorted(extractions_dir.glob("page_*.json")):
		with json_path.open("r", encoding="utf-8") as handle:
			page_data = json.load(handle)

		page_number = page_data.get("page_number")
		for block in page_data.get("content_blocks", []):
			block_index = block.get("block_index")
			for diagram in block.get("diagrams", []):
				diagrams.append(
					{
						"page_number": page_number,
						"block_index": block_index,
						"diagram_index": diagram.get("diagram_index"),
						"description": diagram.get("description"),
					}
				)

	return diagrams


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Sample random diagrams from cached page extractions."
	)
	parser.add_argument(
		"--count",
		type=int,
		default=25,
		help="Number of random diagrams to output (default: 25).",
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
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("random_diagrams_sample.csv"),
		help="CSV file to write the sample to (default: random_diagrams_sample.csv).",
	)
	args = parser.parse_args()

	diagrams = collect_diagrams(args.extractions_dir)
	if not diagrams:
		raise SystemExit(f"No diagrams found in {args.extractions_dir}")

	if args.seed is not None:
		random.seed(args.seed)

	sample_size = min(args.count, len(diagrams))
	sample = random.sample(diagrams, sample_size)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	with args.output.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.writer(handle)
		writer.writerow(["page_number", "block_index", "diagram_index", "description"])
		for diagram in sample:
			writer.writerow(
				[
					diagram["page_number"],
					diagram["block_index"],
					diagram["diagram_index"],
					diagram["description"],
				]
			)

	print(f"Wrote {sample_size} rows to {args.output}")


if __name__ == "__main__":
	main()
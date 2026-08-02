from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


def collect_blocks(extractions_dir: Path) -> list[dict[str, object]]:
	blocks: list[dict[str, object]] = []

	for json_path in sorted(extractions_dir.glob("page_*.json")):
		with json_path.open("r", encoding="utf-8") as handle:
			page_data = json.load(handle)

		page_number = page_data.get("page_number")
		for block in page_data.get("content_blocks", []):
			blocks.append(
				{
					"page_number": page_number,
					"block_index": block.get("block_index"),
					"content_type": block.get("content_type"),
				}
			)

	return blocks


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Sample random content blocks from cached page extractions."
	)
	parser.add_argument(
		"--count",
		type=int,
		default=50,
		help="Number of random blocks to output (default: 50).",
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

	blocks = collect_blocks(args.extractions_dir)
	if not blocks:
		raise SystemExit(f"No blocks found in {args.extractions_dir}")

	if args.seed is not None:
		random.seed(args.seed)

	sample_size = min(args.count, len(blocks))
	sample = random.sample(blocks, sample_size)

	writer = csv.writer(sys.stdout)
	writer.writerow(["page_number", "block_index", "content_type"])
	for block in sample:
		writer.writerow([block["page_number"], block["block_index"], block["content_type"]])


if __name__ == "__main__":
	main()

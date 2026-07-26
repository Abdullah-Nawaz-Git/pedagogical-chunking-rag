"""QA-dataset preparation experiment.

A thin entry point, matching the other files under ``experiments/``: all logic
lives in ``ragkit.qa`` and is driven by the CLI arguments parsed there.

Run the whole pipeline end-to-end with the offline mock provider (no
credentials, fully deterministic):

    python -m experiments.qa_dataset all --provider mock

Or run one stage at a time:

    python -m experiments.qa_dataset select
    python -m experiments.qa_dataset generate --provider vertex
    python -m experiments.qa_dataset validate
    python -m experiments.qa_dataset finalize
    python -m experiments.qa_dataset map-gold

See ``--help`` for the full set of options (``--config``, ``--output-dir``,
``--seed``, ``--force``, ``--dry-run``).
"""

from __future__ import annotations

import sys

from ragkit.qa.runner import main

if __name__ == "__main__":
    sys.exit(main())

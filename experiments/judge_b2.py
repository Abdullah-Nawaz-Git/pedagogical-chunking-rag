"""Thin LLM-as-judge entry point for the structured fixed-window control."""

from ragkit import config as cfg
from ragkit.judge.runner import run


if __name__ == "__main__":
    raise SystemExit(run(cfg.JUDGE_SYSTEM_B2))

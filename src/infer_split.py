"""
Backward-compatible wrapper for the CLI module.

New canonical entrypoint:
  python -m src.cli.infer_split ...
"""

from src.cli.infer_split import main


if __name__ == "__main__":
    main()


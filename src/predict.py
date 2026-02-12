"""
Backward-compatible wrapper for the CLI module.

New canonical entrypoint:
  python -m src.cli.predict ...
"""

from src.cli.predict import main


if __name__ == "__main__":
    main()

"""Console entry point for the LeGen CLI."""
from __future__ import annotations

import sys
from typing import Sequence

from legen import main as _run_main


def main(argv: Sequence[str] | None = None) -> None:
    """Run the packaged LeGen CLI and propagate its exit code."""
    exit_code = _run_main(argv)
    if exit_code not in (None, 0):
        sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()

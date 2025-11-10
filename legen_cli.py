"""Console entry point for the LeGen CLI."""
from __future__ import annotations

import importlib
import sys
from typing import Sequence

_LEGACY_WARNING = (
    "Warning: running legacy LeGen CLI fallback because the installed package "
    "does not expose a 'main' entrypoint. Please upgrade to the latest release."
)


def _restore_sys_argv(original: list[str] | None) -> None:
    if original is not None:
        sys.argv = original


def main(argv: Sequence[str] | None = None) -> None:
    """Run the packaged LeGen CLI and propagate its exit code."""

    forwarded_args = list(argv) if argv is not None else None
    original_sys_argv: list[str] | None = None

    if forwarded_args is not None:
        # Mirror invocation semantics for callers that provide their own argv.
        original_sys_argv = sys.argv.copy()
        program_name = original_sys_argv[0] if original_sys_argv else "legen"
        sys.argv = [program_name, *map(str, forwarded_args)]

    try:
        module = importlib.import_module("legen")
    except Exception:
        _restore_sys_argv(original_sys_argv)
        raise

    entrypoint = getattr(module, "main", None)
    if callable(entrypoint):
        try:
            exit_code = entrypoint(forwarded_args)
        finally:
            _restore_sys_argv(original_sys_argv)

        if exit_code not in (None, 0):
            sys.exit(exit_code)
        return

    _restore_sys_argv(original_sys_argv)

    # Legacy fallback: old packaged versions executed their CLI logic on import.
    # At this point the module has already run using sys.argv, so avoid crashing
    # with ImportError and exit gracefully instead.
    print(_LEGACY_WARNING, file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()

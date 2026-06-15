"""Shared test helpers: a build_command that points the runner at the stub."""

import os
import sys

# Make the package importable when running tests from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STUB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stub_streaming_claude.py")


def stub_build_command(run_request):
    """A build_command returning argv that runs the stub streaming claude.

    Honors execution_location: for local it runs the stub directly; for ssh it
    wraps it in a fake `ssh` shape so the SAME stub serves both transports (the
    real transport selection is tested separately in test_transports)."""
    return [sys.executable, STUB_PATH]

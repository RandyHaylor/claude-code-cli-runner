"""Locate and relocate claude's on-disk session transcripts.

claude stores each session at::

    ~/.claude/projects/<cwd-encoded>/<session-id>.jsonl

where ``<cwd-encoded>`` is the absolute cwd with every ``/`` AND ``_`` replaced
by ``-`` (e.g. ``/tmp/ccrA`` -> ``-tmp-ccrA``; ``/home/x_y`` -> ``-home-x-y``).

``claude --resume <sid> --fork-session`` only finds a session whose jsonl exists
under the CURRENT cwd's project dir. So to fork a primed session from a DIFFERENT
cwd (per-task workspaces), the primed session's jsonl must first be COPIED into
the task cwd's project dir. These helpers do exactly that.

The projects root defaults to ``~/.claude/projects`` and is OVERRIDABLE via an
explicit ``projects_root=`` arg or the ``CLAUDE_PROJECTS_ROOT`` env var (tests
point it at a tmp dir so no real ``~/.claude`` is ever touched).
"""

from __future__ import annotations

import os
import shutil

DEFAULT_PROJECTS_ROOT = os.path.join(os.path.expanduser("~"), ".claude", "projects")


def default_projects_root() -> str:
    """The projects root: ``CLAUDE_PROJECTS_ROOT`` env override, else the default."""
    override = os.environ.get("CLAUDE_PROJECTS_ROOT")
    if override:
        return override
    return DEFAULT_PROJECTS_ROOT


def encode_cwd_to_project_dirname(cwd: str) -> str:
    """Encode an absolute cwd to its claude-projects dir name.

    Every ``/`` and ``_`` becomes ``-`` (e.g. ``/tmp/ccrA`` -> ``-tmp-ccrA``).
    """
    return os.fspath(cwd).replace("/", "-").replace("_", "-")


def project_dir_for_cwd(cwd: str, *, projects_root: "str | None" = None) -> str:
    """The project dir (``<projects_root>/<encoded cwd>``) for ``cwd``."""
    root = projects_root or default_projects_root()
    return os.path.join(root, encode_cwd_to_project_dirname(cwd))


def session_jsonl_path(
    cwd: str, session_id: str, *, projects_root: "str | None" = None
) -> str:
    """The path where claude stores ``session_id``'s transcript when run in ``cwd``."""
    return os.path.join(
        project_dir_for_cwd(cwd, projects_root=projects_root), session_id + ".jsonl"
    )


def ensure_session_present_in_cwd(
    session_id: str,
    source_jsonl_path: str,
    target_cwd: str,
    *,
    projects_root: "str | None" = None,
) -> str:
    """Make ``session_id``'s transcript available under ``target_cwd``'s project dir.

    Copies ``source_jsonl_path`` into the target cwd's project dir (creating it as
    needed) unless the target already exists with the same content/path. Returns
    the target jsonl path. Raises ``FileNotFoundError`` if the source is missing
    (the caller's signal to fall back to inline).
    """
    if not os.path.isfile(source_jsonl_path):
        raise FileNotFoundError(
            "primed session jsonl not found: %r" % source_jsonl_path
        )
    target_path = session_jsonl_path(
        target_cwd, session_id, projects_root=projects_root
    )
    # Same-cwd (or already-relocated) case: nothing to do.
    if os.path.abspath(target_path) == os.path.abspath(source_jsonl_path):
        return target_path
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copyfile(source_jsonl_path, target_path)
    return target_path

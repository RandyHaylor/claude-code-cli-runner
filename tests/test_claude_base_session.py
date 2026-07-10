"""Slice 5b (raw-1211/1212): the warm, dated BASE session every claude task forks
from. Unit tests of the base-session lifecycle (prime-once, reuse-while-fresh,
regenerate-when-stale-or-missing) with injected callables — no claude, no forking here.
"""

import os
import sys
import tempfile

_THIS_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _THIS_REPO not in sys.path:
    sys.path.insert(0, _THIS_REPO)

from claude_code_cli_runner import claude_base_session as base


class _PrimeRecorder:
    """Stands in for the claude priming run: records each primed id and creates the
    transcript file at the path the store will record, so freshness checks pass."""

    def __init__(self, transcript_dir):
        self.transcript_dir = transcript_dir
        self.primed_ids = []

    def jsonl_path_for(self, base_session_id):
        return os.path.join(self.transcript_dir, base_session_id + ".jsonl")

    def prime(self, base_session_id):
        self.primed_ids.append(base_session_id)
        with open(self.jsonl_path_for(base_session_id), "w", encoding="utf-8") as h:
            h.write("{}\n")


def _run(recorder, store_path, now_epoch):
    return base.ensure_fresh_base_session(
        prime_base_session=recorder.prime,
        session_jsonl_path_for=recorder.jsonl_path_for,
        store_path=store_path,
        now_epoch=now_epoch,
    )


def test_first_call_primes_and_records():
    with tempfile.TemporaryDirectory() as d:
        rec = _PrimeRecorder(d)
        store = os.path.join(d, "base.json")
        result = _run(rec, store, now_epoch=1000.0)
        assert len(rec.primed_ids) == 1
        assert result["session_id"] == rec.primed_ids[0]
        assert result["created_epoch"] == 1000.0
        assert os.path.isfile(result["source_jsonl"])


def test_second_call_while_fresh_reuses_no_reprime():
    with tempfile.TemporaryDirectory() as d:
        rec = _PrimeRecorder(d)
        store = os.path.join(d, "base.json")
        first = _run(rec, store, now_epoch=1000.0)
        # One day later — well under the 5-day default.
        second = _run(rec, store, now_epoch=1000.0 + 24 * 3600)
        assert len(rec.primed_ids) == 1  # NOT re-primed
        assert second["session_id"] == first["session_id"]


def test_stale_base_is_regenerated():
    with tempfile.TemporaryDirectory() as d:
        rec = _PrimeRecorder(d)
        store = os.path.join(d, "base.json")
        first = _run(rec, store, now_epoch=1000.0)
        # Six days later — past the 5-day default max age.
        second = _run(rec, store, now_epoch=1000.0 + 6 * 24 * 3600)
        assert len(rec.primed_ids) == 2  # re-primed
        assert second["session_id"] != first["session_id"]


def test_missing_transcript_forces_reprime():
    with tempfile.TemporaryDirectory() as d:
        rec = _PrimeRecorder(d)
        store = os.path.join(d, "base.json")
        first = _run(rec, store, now_epoch=1000.0)
        os.remove(first["source_jsonl"])  # transcript vanished (cwd wiped)
        second = _run(rec, store, now_epoch=1000.0 + 60)
        assert len(rec.primed_ids) == 2
        assert second["session_id"] != first["session_id"]


def test_max_age_env_override(monkeypatch=None):
    with tempfile.TemporaryDirectory() as d:
        os.environ["UNHARNESS_CLAUDE_BASE_SESSION_MAX_AGE_DAYS"] = "1"
        try:
            rec = _PrimeRecorder(d)
            store = os.path.join(d, "base.json")
            _run(rec, store, now_epoch=1000.0)
            # 2 days later, with a 1-day cap -> regenerate.
            _run(rec, store, now_epoch=1000.0 + 2 * 24 * 3600)
            assert len(rec.primed_ids) == 2
        finally:
            del os.environ["UNHARNESS_CLAUDE_BASE_SESSION_MAX_AGE_DAYS"]


def test_forget_base_session_clears_store():
    with tempfile.TemporaryDirectory() as d:
        rec = _PrimeRecorder(d)
        store = os.path.join(d, "base.json")
        _run(rec, store, now_epoch=1000.0)
        base.forget_base_session(store_path=store)
        _run(rec, store, now_epoch=1000.0 + 60)
        assert len(rec.primed_ids) == 2  # primed again after forget


def test_fork_enabled_by_default_and_env_off_switch():
    from claude_code_cli_runner.runner import _claude_base_session_fork_enabled
    os.environ.pop("UNHARNESS_ENABLE_CLAUDE_BASE_SESSION_FORK", None)
    assert _claude_base_session_fork_enabled() is True
    for off in ("0", "false", "No", "off"):
        os.environ["UNHARNESS_ENABLE_CLAUDE_BASE_SESSION_FORK"] = off
        assert _claude_base_session_fork_enabled() is False
    os.environ["UNHARNESS_ENABLE_CLAUDE_BASE_SESSION_FORK"] = "1"
    assert _claude_base_session_fork_enabled() is True
    os.environ.pop("UNHARNESS_ENABLE_CLAUDE_BASE_SESSION_FORK", None)

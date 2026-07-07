"""Generic -> harness-specific tool-name translation (the runner owns the mapping)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner.harness_tool_name_translation import (
    translate_generic_tool_names_to_harness_specific,
)


def test_claude_vocabulary_maps_identity():
    assert translate_generic_tool_names_to_harness_specific(
        ["Read", "Bash", "Grep"], "claude_cli"
    ) == ["Read", "Bash", "Grep"]


def test_unmapped_generic_name_passes_through_as_identity():
    # A name not in the reference falls back to identity (still functions), so an
    # as-yet-undefined tool is not silently dropped.
    assert translate_generic_tool_names_to_harness_specific(
        ["Read", "SomeFutureTool"], "claude_cli"
    ) == ["Read", "SomeFutureTool"]


def test_unknown_harness_falls_back_to_identity():
    assert translate_generic_tool_names_to_harness_specific(
        ["Read", "Edit"], "no_such_harness"
    ) == ["Read", "Edit"]


def test_empty_list_translates_to_empty_list():
    assert translate_generic_tool_names_to_harness_specific([], "claude_cli") == []

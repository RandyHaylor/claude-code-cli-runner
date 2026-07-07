"""Translate GENERIC (Unharness-level) tool names into HARNESS-specific tool names.

Unharness assigns a session's tools by generic name (a small vocabulary that reuses
claude's tool names). The runner is the layer over the harnesses, so it owns the
conversion: before emitting a harness's tool-restriction flag, it converts each
generic name into that harness's own name via a JSON reference
(``tool_name_conversion_reference.json``). A generic name with no mapping for the
target harness falls back to identity (passed through unchanged) with a warning, so
an as-yet-unmapped tool still functions rather than silently vanishing.
"""

from __future__ import annotations

import json
import os
import sys

_CONVERSION_REFERENCE_FILENAME = "tool_name_conversion_reference.json"
_HARNESS_MAP_KEY = "harness_tool_name_by_generic_name"


def _load_harness_tool_name_maps() -> dict:
    reference_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), _CONVERSION_REFERENCE_FILENAME
    )
    with open(reference_path, "r", encoding="utf-8") as reference_file:
        reference = json.load(reference_file)
    return reference.get(_HARNESS_MAP_KEY, {})


def translate_generic_tool_names_to_harness_specific(
    generic_tool_names: "list[str]", harness_name: str
) -> "list[str]":
    """Convert each generic tool name to its harness-specific name for ``harness_name``.

    Unmapped names (or an unknown harness) fall back to identity — the generic name is
    used as-is and a warning is emitted to stderr — so the session still gets the tool
    while the conversion reference is extended. Order is preserved. Keys beginning with
    an underscore in the reference (notes) are ignored."""
    harness_maps = _load_harness_tool_name_maps()
    this_harness_map = {
        generic: specific
        for generic, specific in (harness_maps.get(harness_name) or {}).items()
        if not generic.startswith("_")
    }
    harness_specific_names = []
    for generic_name in generic_tool_names:
        if generic_name in this_harness_map:
            harness_specific_names.append(this_harness_map[generic_name])
        else:
            sys.stderr.write(
                "harness_tool_name_translation: no %r mapping for generic tool %r; "
                "passing it through unchanged (extend %s)\n"
                % (harness_name, generic_name, _CONVERSION_REFERENCE_FILENAME)
            )
            harness_specific_names.append(generic_name)
    return harness_specific_names

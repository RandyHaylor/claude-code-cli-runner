"""Multimodal INPUT serialization: image + document blocks -> stream-json."""

import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import DocumentBlock, ImageBlock, TextBlock
from claude_code_cli_runner.content import build_user_message, serialize_block


def test_text_block_serialization():
    entry = serialize_block(TextBlock(text="hi"))
    assert entry == {"type": "text", "text": "hi"}


def test_image_block_from_path(tmp_path):
    image_path = tmp_path / "pic.png"
    image_path.write_bytes(b"\x89PNGdata")
    entry = serialize_block(ImageBlock(mime_type="image/png", path=str(image_path)))
    assert entry["type"] == "image"
    assert entry["source"]["media_type"] == "image/png"
    assert entry["source"]["data"] == base64.b64encode(b"\x89PNGdata").decode("ascii")


def test_document_block_inline_base64():
    data = base64.b64encode(b"%PDF-1.4").decode("ascii")
    entry = serialize_block(
        DocumentBlock(mime_type="application/pdf", data_base64=data, name="spec.pdf")
    )
    assert entry["type"] == "document"
    assert entry["source"]["media_type"] == "application/pdf"
    assert entry["source"]["data"] == data
    assert entry["title"] == "spec.pdf"


def test_build_user_message_assembles_content_array(tmp_path):
    image_path = tmp_path / "pic.png"
    image_path.write_bytes(b"img")
    message = build_user_message(
        [
            TextBlock(text="describe this"),
            ImageBlock(mime_type="image/png", path=str(image_path)),
        ]
    )
    assert message["type"] == "user"
    content = message["message"]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image"

"""Tests for strict ACDC Info.cfg metadata parsing."""

from pathlib import Path

import pytest

from cardiac_pathology.data import AcdcMetadataParser


def write_info_cfg(tmp_path: Path, text: str) -> Path:
    """Write temporary metadata text for parser logic tests."""
    info_path = tmp_path / "Info.cfg"
    info_path.write_text(text, encoding="utf-8")
    return info_path


def test_valid_metadata_parsing(tmp_path: Path) -> None:
    """Parser accepts valid ED/ES/Group metadata."""
    info_path = write_info_cfg(
        tmp_path,
        "ED: 1\nES: 12\nGroup: NOR\nHeight: 180\n",
    )

    metadata = AcdcMetadataParser().parse(info_path)

    assert metadata.ed_frame == 1
    assert metadata.es_frame == 12
    assert metadata.group == "NOR"
    assert metadata.values["Height"] == "180"


def test_missing_required_key(tmp_path: Path) -> None:
    """Parser rejects missing ED/ES/Group keys."""
    info_path = write_info_cfg(tmp_path, "ED: 1\nGroup: NOR\n")

    with pytest.raises(ValueError, match="missing required"):
        AcdcMetadataParser().parse(info_path)


def test_duplicate_key(tmp_path: Path) -> None:
    """Parser rejects duplicate keys."""
    info_path = write_info_cfg(tmp_path, "ED: 1\nED: 2\nES: 12\nGroup: NOR\n")

    with pytest.raises(ValueError, match="duplicate key"):
        AcdcMetadataParser().parse(info_path)


def test_malformed_line(tmp_path: Path) -> None:
    """Parser rejects non-empty lines without a colon."""
    info_path = write_info_cfg(tmp_path, "ED: 1\nbad line\nES: 12\nGroup: NOR\n")

    with pytest.raises(ValueError, match="malformed line"):
        AcdcMetadataParser().parse(info_path)


@pytest.mark.parametrize("key,value", [("ED", "0"), ("ES", "-1"), ("ED", "abc")])
def test_invalid_ed_es(tmp_path: Path, key: str, value: str) -> None:
    """Parser rejects non-positive or non-integer ED/ES values."""
    values = {"ED": "1", "ES": "12", "Group": "NOR"}
    values[key] = value
    info_path = write_info_cfg(
        tmp_path,
        f"ED: {values['ED']}\nES: {values['ES']}\nGroup: {values['Group']}\n",
    )

    with pytest.raises(ValueError, match=key):
        AcdcMetadataParser().parse(info_path)

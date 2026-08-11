"""Parser for ACDC patient Info.cfg metadata files."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

REQUIRED_KEYS = frozenset({"ED", "ES", "Group"})


@dataclass(frozen=True, slots=True)
class AcdcMetadata:
    """Parsed ACDC Info.cfg metadata required for classification indexing."""

    ed_frame: int
    es_frame: int
    group: str
    values: Mapping[str, str]


class AcdcMetadataParser:
    """Strict parser for real ACDC Info.cfg files."""

    def parse(self, info_path: Path) -> AcdcMetadata:
        """Parse and validate one Info.cfg file."""
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing Info.cfg: {info_path}")

        values: dict[str, str] = {}
        with info_path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if ":" not in line:
                    raise ValueError(
                        f"{info_path}: malformed line {line_number}: {raw_line.rstrip()}"
                    )

                key, value = line.split(":", maxsplit=1)
                key = key.strip()
                value = value.strip()
                if not key:
                    raise ValueError(f"{info_path}: empty key on line {line_number}")
                if key in values:
                    raise ValueError(f"{info_path}: duplicate key {key!r}")
                values[key] = value

        missing_keys = REQUIRED_KEYS - values.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"{info_path}: missing required key(s): {missing}")

        ed_frame = self._parse_positive_frame(info_path, "ED", values["ED"])
        es_frame = self._parse_positive_frame(info_path, "ES", values["ES"])
        group = values["Group"].strip()
        if not group:
            raise ValueError(f"{info_path}: Group must be non-empty")

        return AcdcMetadata(
            ed_frame=ed_frame,
            es_frame=es_frame,
            group=group,
            values=MappingProxyType(dict(values)),
        )

    def _parse_positive_frame(self, info_path: Path, key: str, value: str) -> int:
        """Parse ED/ES frame value as a positive integer."""
        try:
            frame = int(value)
        except ValueError as error:
            raise ValueError(f"{info_path}: {key} must be a positive integer") from error

        if frame < 1:
            raise ValueError(f"{info_path}: {key} must be >= 1")
        return frame

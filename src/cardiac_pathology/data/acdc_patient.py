"""Immutable ACDC patient record for classification indexing."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AcdcPatient:
    """Patient-level ED/ES classification sample metadata and paths."""

    patient_id: str
    class_index: int
    class_name: str
    ed_frame: int
    es_frame: int
    ed_path: Path
    es_path: Path
    cine_4d_path: Path
    info_path: Path

    def __post_init__(self) -> None:
        """Validate immutable patient metadata invariants."""
        if not self.patient_id:
            raise ValueError("patient_id must be non-empty")
        if self.class_index < 0:
            raise ValueError(f"{self.patient_id}: class_index must be non-negative")
        if not self.class_name:
            raise ValueError(f"{self.patient_id}: class_name must be non-empty")
        if self.ed_frame < 1:
            raise ValueError(f"{self.patient_id}: ED frame must be >= 1")
        if self.es_frame < 1:
            raise ValueError(f"{self.patient_id}: ES frame must be >= 1")
        if self.ed_frame == self.es_frame:
            raise ValueError(f"{self.patient_id}: ED and ES frames must differ")

        for path_name, path in [
            ("ED", self.ed_path),
            ("ES", self.es_path),
            ("4D cine", self.cine_4d_path),
            ("Info.cfg", self.info_path),
        ]:
            if not path.is_file():
                raise FileNotFoundError(f"{self.patient_id}: missing {path_name} file: {path}")

"""Validate production preprocessing against the frozen Phase 1 contract."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cardiac_pathology.data import AcdcDatasetIndexer  # noqa: E402
from cardiac_pathology.preprocessing import PatientPreprocessor, PreprocessingConfig  # noqa: E402

DATASET_DIR = REPOSITORY_ROOT / "data/raw/acdc/training"
CONFIG_PATH = REPOSITORY_ROOT / "configs/default.yaml"
CLASS_MAPPING_PATH = REPOSITORY_ROOT / "configs/class_mapping.json"
PHASE1_REFERENCE_CSV = (
    REPOSITORY_ROOT / "artifacts/dataset_inspection/final_preprocessing_validation.csv"
)
OUTPUT_DIR = REPOSITORY_ROOT / "artifacts/preprocessing_validation"
OUTPUT_CSV = OUTPUT_DIR / "phase3_validation.csv"
OUTPUT_MD = OUTPUT_DIR / "phase3_validation.md"


def main() -> None:
    """Run real-data preprocessing validation and write artifacts."""
    config = preprocessing_config_from_yaml(CONFIG_PATH)
    patients = AcdcDatasetIndexer(DATASET_DIR, CLASS_MAPPING_PATH).index_patients()
    preprocessor = PatientPreprocessor(config)
    records: list[dict[str, Any]] = []
    failures: list[str] = []

    for patient in patients:
        try:
            result = preprocessor.preprocess(patient)
            records.append(
                {
                    "patient_id": patient.patient_id,
                    "class": patient.class_name,
                    "source_shape": str(result.source_shape_xyz),
                    "source_spacing": str(result.source_spacing_xyz),
                    "resampled_shape": str(result.resampled_shape_xyz),
                    "resampled_size_z": result.resampled_shape_xyz[2],
                    "valid_depth": result.valid_depth,
                    "z_padding_lower": result.z_padding_lower,
                    "z_padding_upper": result.z_padding_upper,
                    "z_padding_total": result.z_padding_lower + result.z_padding_upper,
                    "clip_lower": result.clip_lower,
                    "clip_upper": result.clip_upper,
                    "normalization_mean": result.normalization_mean,
                    "normalization_std": result.normalization_std,
                    "output_shape": str(tuple(result.tensor.shape)),
                    "output_dtype": str(result.tensor.dtype),
                    "output_contiguous": bool(result.tensor.flags.c_contiguous),
                    "all_finite": bool(np.isfinite(result.tensor).all()),
                    "target_spacing": str(result.target_spacing_xyz),
                    "no_xy_padding": result.crop_end_xy[0] - result.crop_start_xy[0] == 144
                    and result.crop_end_xy[1] - result.crop_start_xy[1] == 144,
                    "valid_depth_leq_14": result.valid_depth <= 14,
                    "no_z_cropping": result.valid_depth <= 14,
                    "normalization_std_gt_epsilon": result.normalization_std > config.epsilon,
                }
            )
        except Exception as error:
            failures.append(f"{patient.patient_id}: {error}")

    dataframe = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(OUTPUT_CSV, index=False)

    phase1_parity = compare_phase1_reference(dataframe)
    write_markdown_report(dataframe, failures, phase1_parity)
    print_report(dataframe, failures, phase1_parity)

    if failures:
        raise RuntimeError(f"{len(failures)} preprocessing failure(s)")


def preprocessing_config_from_yaml(path: Path) -> PreprocessingConfig:
    """Construct PreprocessingConfig from configs/default.yaml."""
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    preprocessing = config["preprocessing"]
    target_spacing = preprocessing["target_spacing_mm"]
    target_shape = preprocessing["target_shape"]
    intensity = preprocessing["intensity"]
    return PreprocessingConfig(
        target_orientation=preprocessing["orientation"],
        target_spacing_xyz=(
            float(target_spacing["x"]),
            float(target_spacing["y"]),
            float(target_spacing["z"]),
        ),
        target_shape_dhw=(
            int(target_shape["d"]),
            int(target_shape["h"]),
            int(target_shape["w"]),
        ),
        lower_percentile=float(intensity["lower_percentile"]),
        upper_percentile=float(intensity["upper_percentile"]),
        epsilon=float(intensity["epsilon"]),
        z_padding_value=float(preprocessing["padding"]["z_value"]),
    )


def compare_phase1_reference(dataframe: pd.DataFrame) -> str:
    """Compare patient-level depth and padding metadata to Phase 1 artifact."""
    if not PHASE1_REFERENCE_CSV.is_file():
        return "Phase 1 patient-level reference artifact: unavailable"

    reference = pd.read_csv(PHASE1_REFERENCE_CSV)
    columns = [
        "patient_id",
        "resampled_size_z",
        "z_padding_lower",
        "z_padding_upper",
        "z_padding_total",
    ]
    merged = dataframe[columns].merge(
        reference[columns],
        on="patient_id",
        suffixes=("_phase3", "_phase1"),
        how="inner",
    )
    if len(merged) != len(dataframe):
        return "FAIL: Phase 1 parity patient set mismatch"
    for column in columns[1:]:
        if not (merged[f"{column}_phase3"] == merged[f"{column}_phase1"]).all():
            return f"FAIL: Phase 1 parity mismatch for {column}"
    return "PASS: Phase 1 patient-level depth/padding parity"


def write_markdown_report(
    dataframe: pd.DataFrame,
    failures: list[str],
    phase1_parity: str,
) -> None:
    """Write concise preprocessing validation Markdown report."""
    report = [
        "# Phase 3 Preprocessing Validation",
        "",
        f"Patients processed: {len(dataframe)}",
        f"Failed patients: {len(failures)}",
        "",
        "## Valid Depth",
        six_number_summary(dataframe["valid_depth"]).to_markdown(),
        "",
        "## Z Padding",
        six_number_summary(dataframe["z_padding_total"]).to_markdown(),
        "",
        "## Phase 1 Patient-Level Parity",
        phase1_parity,
        "",
        "## Output Validation",
        output_validation(dataframe).to_markdown(),
        "",
        "## Failures",
        "None." if not failures else "\n".join(failures),
    ]
    OUTPUT_MD.write_text("\n".join(report) + "\n", encoding="utf-8")


def print_report(dataframe: pd.DataFrame, failures: list[str], phase1_parity: str) -> None:
    """Print concise validation report."""
    print("Phase 3 preprocessing validation")
    print(f"\nPatients processed: {len(dataframe)}")
    print(f"Failed patients: {len(failures)}")
    print("\nValid-depth summary:")
    print(six_number_summary(dataframe["valid_depth"]).to_string())
    print("\nZ-padding summary:")
    print(six_number_summary(dataframe["z_padding_total"]).to_string())
    print(f"\n{phase1_parity}")
    print("\nOutput validation:")
    print(output_validation(dataframe).to_string())
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved report: {OUTPUT_MD}")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  {failure}")


def six_number_summary(series: pd.Series) -> pd.Series:
    """Return min/p05/median/mean/p95/max."""
    return pd.Series(
        {
            "min": series.min(),
            "p05": series.quantile(0.05),
            "median": series.median(),
            "mean": series.mean(),
            "p95": series.quantile(0.95),
            "max": series.max(),
        }
    ).round(4)


def output_validation(dataframe: pd.DataFrame) -> pd.Series:
    """Summarize tensor output contract validation."""
    return pd.Series(
        {
            "correct_shape": int((dataframe["output_shape"] == "(2, 14, 144, 144)").sum()),
            "float32": int((dataframe["output_dtype"] == "float32").sum()),
            "contiguous": int(dataframe["output_contiguous"].sum()),
            "finite": int(dataframe["all_finite"].sum()),
            "target_spacing_expected": int(
                (dataframe["target_spacing"] == "(1.5, 1.5, 7.5)").sum()
            ),
            "no_xy_padding": int(dataframe["no_xy_padding"].sum()),
            "valid_depth_leq_14": int(dataframe["valid_depth_leq_14"].sum()),
            "no_z_cropping": int(dataframe["no_z_cropping"].sum()),
            "normalization_std_gt_epsilon": int(dataframe["normalization_std_gt_epsilon"].sum()),
        }
    )


if __name__ == "__main__":
    main()

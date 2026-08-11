"""Validate the frozen Phase 1 preprocessing candidate on real ACDC data."""

import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cardiac_pathology_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_SPACING_MM = (1.50, 1.50, 7.50)
TARGET_XY_PIXELS = 144
TARGET_DEPTH = 14
TARGET_SHAPE = (2, TARGET_DEPTH, TARGET_XY_PIXELS, TARGET_XY_PIXELS)
NORMALIZATION_LOWER_PERCENTILE = 0.5
NORMALIZATION_UPPER_PERCENTILE = 99.5
NORMALIZATION_EPSILON = 1e-6
VALIDATION_TOLERANCE = 1e-5


class FinalPreprocessingCandidateAnalyzer:
    """Run final preprocessing-candidate validation across all ACDC patients."""

    def __init__(
        self,
        dataset_dir: Path,
        class_mapping_path: Path,
        output_dir: Path,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.class_mapping_path = class_mapping_path
        self.output_dir = output_dir
        self.output_csv = output_dir / "final_preprocessing_validation.csv"
        self.output_report = output_dir / "final_preprocessing_validation.md"
        self.qa_dir = output_dir / "final_preprocessing_qa"
        self.qa_contact_sheet = output_dir / "final_preprocessing_qa_contact_sheet.png"
        self.qa_manifest = output_dir / "final_preprocessing_qa_manifest.txt"
        self.validation_failures: list[str] = []
        self.residual_shortcut_metrics: dict[str, float] = {}

    def run(self) -> dict[str, Any]:
        """Validate preprocessing, write artifacts, and return final summaries."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.qa_dir.mkdir(parents=True, exist_ok=True)

        patient_dirs = sorted(self.dataset_dir.glob("patient*"))
        if not patient_dirs:
            raise RuntimeError(f"No patient directories found in {self.dataset_dir}")

        records = [self._process_patient(patient_dir, keep_tensor=False) for patient_dir in patient_dirs]
        dataframe = pd.DataFrame(records)
        dataframe.to_csv(self.output_csv, index=False)

        class_order = self._load_class_order()
        self.residual_shortcut_metrics = self._run_residual_padding_baseline(
            dataframe,
            class_order,
        )
        qa_entries = self._generate_qa_package(dataframe)
        summary = self._build_summary(dataframe)
        self._write_report(summary, qa_entries)

        return {
            "dataframe": dataframe,
            "summary": summary,
            "residual_shortcut_metrics": self.residual_shortcut_metrics,
            "qa_entries": qa_entries,
            "output_csv": self.output_csv,
            "output_report": self.output_report,
            "qa_contact_sheet": self.qa_contact_sheet,
            "qa_manifest": self.qa_manifest,
            "validation_failures": self.validation_failures,
        }

    def print_final_summary(self, results: dict[str, Any]) -> None:
        """Print the required run summary."""
        summary = results["summary"]
        print("Final preprocessing validation")
        print("\nActual resampled-depth summary")
        print(summary["resampled_depth"].to_string())
        print("\nZ-padding summary")
        print(summary["z_padding"].to_string())
        print("\nZ-padding by class")
        print(summary["z_padding_by_class"].to_string())
        print("\nClipping-bound summary")
        print(summary["clipping_bounds"].to_string())
        print("\nNormalization validation")
        print(summary["normalization_validation"].to_string())
        print("\nFinal output validation")
        print(summary["final_output_validation"].to_string())
        metrics = results["residual_shortcut_metrics"]
        print("\nResidual padding shortcut diagnostic")
        print(f"OOF Accuracy: {metrics['accuracy']:.3f}")
        print(f"OOF Macro F1: {metrics['macro_f1']:.3f}")
        print("Chance accuracy: 0.200")
        with Image.open(results["qa_contact_sheet"]) as image:
            dimensions = f"{image.width}x{image.height}"
        print("\nQA")
        print(f"contact_sheet: {results['qa_contact_sheet']}")
        print(f"contact_sheet_dimensions: {dimensions}")
        print("\nGenerated paths")
        print(f"csv: {results['output_csv']}")
        print(f"report: {results['output_report']}")
        print(f"qa_manifest: {results['qa_manifest']}")
        if results["validation_failures"]:
            print("\nValidation failures")
            for failure in results["validation_failures"]:
                print(f"  {failure}")

    def _process_patient(self, patient_dir: Path, keep_tensor: bool) -> dict[str, Any]:
        """Load, preprocess, validate, and summarize one patient."""
        patient_id = patient_dir.name
        metadata = self._parse_info_file(patient_dir / "Info.cfg")
        ed_frame = int(metadata["ED"])
        es_frame = int(metadata["ES"])
        class_name = metadata["Group"]

        ed_image = nib.load(patient_dir / f"{patient_id}_frame{ed_frame:02d}.nii.gz")
        es_image = nib.load(patient_dir / f"{patient_id}_frame{es_frame:02d}.nii.gz")
        self._validate_source_images(patient_id, ed_image, es_image)

        ed_source = np.asanyarray(ed_image.dataobj)
        es_source = np.asanyarray(es_image.dataobj)
        source_spacing = self._spatial_spacing(ed_image)
        resampled_shape = self._target_size_from_center_span(ed_image.shape, source_spacing)
        ed_resampled = self._resample_to_target_grid(ed_source, source_spacing, resampled_shape)
        es_resampled = self._resample_to_target_grid(es_source, source_spacing, resampled_shape)

        if ed_resampled.shape != es_resampled.shape:
            raise ValueError(f"{patient_id}: ED/ES resampled shapes differ")

        ed_cropped, xy_operation = self._center_crop_xy(ed_resampled, patient_id)
        es_cropped, es_xy_operation = self._center_crop_xy(es_resampled, patient_id)
        if xy_operation != es_xy_operation:
            raise ValueError(f"{patient_id}: ED/ES XY crop geometry differs")

        if xy_operation["pad_x_total"] > 0 or xy_operation["pad_y_total"] > 0:
            raise ValueError(f"{patient_id}: XY padding required after resampling")

        real_depth = ed_cropped.shape[2]
        if real_depth > TARGET_DEPTH:
            raise ValueError(
                f"{patient_id}: resampled depth {real_depth} exceeds target D={TARGET_DEPTH}"
            )

        normalized = self._joint_normalize(ed_cropped, es_cropped, patient_id)
        ed_normalized = normalized["ed_normalized"]
        es_normalized = normalized["es_normalized"]
        ed_padded, padding = self._center_pad_z(ed_normalized, patient_id)
        es_padded, es_padding = self._center_pad_z(es_normalized, patient_id)
        if padding != es_padding:
            raise ValueError(f"{patient_id}: ED/ES Z padding differs")

        ed_final = np.transpose(ed_padded, (2, 1, 0))
        es_final = np.transpose(es_padded, (2, 1, 0))
        tensor = np.ascontiguousarray(np.stack([ed_final, es_final], axis=0).astype(np.float32))
        self._validate_tensor(patient_id, tensor)

        record: dict[str, Any] = {
            "patient_id": patient_id,
            "class": class_name,
            "source_size_x": ed_image.shape[0],
            "source_size_y": ed_image.shape[1],
            "source_size_z": ed_image.shape[2],
            "source_spacing_x_mm": source_spacing[0],
            "source_spacing_y_mm": source_spacing[1],
            "source_spacing_z_mm": source_spacing[2],
            "resampled_size_x": ed_resampled.shape[0],
            "resampled_size_y": ed_resampled.shape[1],
            "resampled_size_z": ed_resampled.shape[2],
            "clip_lower": normalized["clip_lower"],
            "clip_upper": normalized["clip_upper"],
            "normalization_mean": normalized["normalization_mean"],
            "normalization_std": normalized["normalization_std"],
            "ed_valid_mean_after_normalization": float(ed_normalized.mean()),
            "ed_valid_std_after_normalization": float(ed_normalized.std()),
            "es_valid_mean_after_normalization": float(es_normalized.mean()),
            "es_valid_std_after_normalization": float(es_normalized.std()),
            "joint_valid_mean_after_normalization": normalized["joint_mean_after"],
            "joint_valid_std_after_normalization": normalized["joint_std_after"],
            "z_padding_lower": padding["lower"],
            "z_padding_upper": padding["upper"],
            "z_padding_total": padding["total"],
            "z_padding_fraction": padding["total"] / TARGET_DEPTH,
            "output_shape": str(tuple(tensor.shape)),
            "output_dtype": str(tensor.dtype),
            "output_contiguous": bool(tensor.flags.c_contiguous),
            "has_nan": bool(np.isnan(tensor).any()),
            "has_posinf": bool(np.isposinf(tensor).any()),
            "has_neginf": bool(np.isneginf(tensor).any()),
            "xy_padding_x_total": xy_operation["pad_x_total"],
            "xy_padding_y_total": xy_operation["pad_y_total"],
            "z_cropping_total": 0,
            "normalization_std_gt_epsilon": normalized["normalization_std"] > NORMALIZATION_EPSILON,
            "correct_shape": tensor.shape == TARGET_SHAPE,
            "float32_dtype": tensor.dtype == np.float32,
            "c_contiguous": bool(tensor.flags.c_contiguous),
        }
        if keep_tensor:
            record["_ed_source"] = ed_source
            record["_es_source"] = es_source
            record["_ed_pre_normalization"] = ed_cropped
            record["_es_pre_normalization"] = es_cropped
            record["_tensor"] = tensor
        return record

    def _validate_source_images(
        self,
        patient_id: str,
        ed_image: nib.spatialimages.SpatialImage,
        es_image: nib.spatialimages.SpatialImage,
    ) -> None:
        """Validate standalone ED/ES source image geometry."""
        if len(ed_image.shape) != 3 or len(es_image.shape) != 3:
            raise ValueError(f"{patient_id}: ED/ES images must be 3D")
        if ed_image.shape != es_image.shape:
            raise ValueError(f"{patient_id}: ED/ES shape mismatch")
        if self._spatial_spacing(ed_image) != self._spatial_spacing(es_image):
            raise ValueError(f"{patient_id}: ED/ES spacing mismatch")
        ed_orientation = "".join(nib.aff2axcodes(ed_image.affine))
        es_orientation = "".join(nib.aff2axcodes(es_image.affine))
        if ed_orientation != "LPS" or es_orientation != "LPS":
            raise ValueError(
                f"{patient_id}: expected standalone LPS orientation, got "
                f"ED={ed_orientation}, ES={es_orientation}"
            )

    def _resample_to_target_grid(
        self,
        source: np.ndarray,
        source_spacing: tuple[float, float, float],
        target_shape: tuple[int, int, int],
    ) -> np.ndarray:
        """Linearly resample source to the frozen target-spacing grid."""
        coordinate_axes = []
        for axis, target_size in enumerate(target_shape):
            source_center = (source.shape[axis] - 1) / 2
            target_center = (target_size - 1) / 2
            spacing_ratio = TARGET_SPACING_MM[axis] / source_spacing[axis]
            coordinates = (
                np.arange(target_size, dtype=np.float64) - target_center
            ) * spacing_ratio + source_center
            coordinate_axes.append(coordinates)

        grid = np.meshgrid(*coordinate_axes, indexing="ij")
        return map_coordinates(
            source.astype(np.float64),
            grid,
            order=1,
            mode="nearest",
            prefilter=False,
        )

    def _center_crop_xy(
        self,
        volume: np.ndarray,
        patient_id: str,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Apply deterministic FOV-centered XY crop without padding."""
        size_x, size_y, _size_z = volume.shape
        if size_x < TARGET_XY_PIXELS or size_y < TARGET_XY_PIXELS:
            raise ValueError(
                f"{patient_id}: XY padding required, resampled XY={size_x}x{size_y}"
            )
        crop_x = size_x - TARGET_XY_PIXELS
        crop_y = size_y - TARGET_XY_PIXELS
        x_lower = crop_x // 2
        y_lower = crop_y // 2
        cropped = volume[
            x_lower : x_lower + TARGET_XY_PIXELS,
            y_lower : y_lower + TARGET_XY_PIXELS,
            :,
        ]
        return cropped, {
            "crop_x_lower": x_lower,
            "crop_x_upper": crop_x - x_lower,
            "crop_y_lower": y_lower,
            "crop_y_upper": crop_y - y_lower,
            "pad_x_total": 0,
            "pad_y_total": 0,
        }

    def _joint_normalize(
        self,
        ed_volume: np.ndarray,
        es_volume: np.ndarray,
        patient_id: str,
    ) -> dict[str, Any]:
        """Jointly clip and z-score ED/ES valid voxels before artificial padding."""
        joint_values = np.concatenate([ed_volume.ravel(), es_volume.ravel()])
        clip_lower = float(np.percentile(joint_values, NORMALIZATION_LOWER_PERCENTILE))
        clip_upper = float(np.percentile(joint_values, NORMALIZATION_UPPER_PERCENTILE))
        ed_clipped = np.clip(ed_volume, clip_lower, clip_upper)
        es_clipped = np.clip(es_volume, clip_lower, clip_upper)
        clipped_joint = np.concatenate([ed_clipped.ravel(), es_clipped.ravel()])
        mean = float(clipped_joint.mean())
        std = float(clipped_joint.std())
        if std <= NORMALIZATION_EPSILON:
            raise ValueError(f"{patient_id}: normalization std <= epsilon")
        ed_normalized = (ed_clipped - mean) / max(std, NORMALIZATION_EPSILON)
        es_normalized = (es_clipped - mean) / max(std, NORMALIZATION_EPSILON)
        normalized_joint = np.concatenate([ed_normalized.ravel(), es_normalized.ravel()])
        return {
            "clip_lower": clip_lower,
            "clip_upper": clip_upper,
            "normalization_mean": mean,
            "normalization_std": std,
            "ed_normalized": ed_normalized,
            "es_normalized": es_normalized,
            "joint_mean_after": float(normalized_joint.mean()),
            "joint_std_after": float(normalized_joint.std()),
        }

    def _center_pad_z(
        self,
        volume: np.ndarray,
        patient_id: str,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Center-pad Z to target depth using zero-valued artificial slices."""
        depth = volume.shape[2]
        if depth > TARGET_DEPTH:
            raise ValueError(f"{patient_id}: Z cropping would be required")
        total_padding = TARGET_DEPTH - depth
        lower = total_padding // 2
        upper = total_padding - lower
        padded = np.pad(
            volume,
            ((0, 0), (0, 0), (lower, upper)),
            mode="constant",
            constant_values=0.0,
        )
        return padded, {"lower": lower, "upper": upper, "total": total_padding}

    def _validate_tensor(self, patient_id: str, tensor: np.ndarray) -> None:
        """Validate final tensor shape, dtype, memory layout, and finite values."""
        if tensor.shape != TARGET_SHAPE:
            raise ValueError(f"{patient_id}: output shape {tensor.shape} != {TARGET_SHAPE}")
        if tensor.dtype != np.float32:
            raise ValueError(f"{patient_id}: output dtype {tensor.dtype} != float32")
        if not tensor.flags.c_contiguous:
            raise ValueError(f"{patient_id}: output tensor is not C-contiguous")
        if np.isnan(tensor).any() or np.isinf(tensor).any():
            raise ValueError(f"{patient_id}: output tensor contains NaN or Inf")

    def _run_residual_padding_baseline(
        self,
        dataframe: pd.DataFrame,
        class_order: list[str],
    ) -> dict[str, float]:
        """Run the final residual padding shortcut diagnostic baseline."""
        features = dataframe[["z_padding_fraction", "z_padding_total", "resampled_size_z"]]
        labels = dataframe["class"]
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1_000, random_state=42)),
            ]
        )
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        predictions = cross_val_predict(pipeline, features, labels, cv=folds)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(
                labels,
                predictions,
                labels=class_order,
                average="macro",
            ),
        }

    def _build_summary(self, dataframe: pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
        """Build cohort validation summaries."""
        return {
            "resampled_depth": self._six_number_summary(dataframe["resampled_size_z"]),
            "z_padding": self._six_number_summary(dataframe["z_padding_total"]),
            "z_padding_by_class": dataframe.groupby("class")["z_padding_total"]
            .agg(
                median="median",
                mean="mean",
                p95=lambda values: values.quantile(0.95),
                max="max",
            )
            .round(4),
            "clipping_bounds": pd.DataFrame(
                {
                    "clip_lower": dataframe["clip_lower"].agg(["min", "median", "max"]),
                    "clip_upper": dataframe["clip_upper"].agg(["min", "median", "max"]),
                }
            ).round(4),
            "normalization_validation": pd.Series(
                {
                    "max_abs_joint_mean_deviation": float(
                        dataframe["joint_valid_mean_after_normalization"].abs().max()
                    ),
                    "max_abs_joint_std_deviation": float(
                        (dataframe["joint_valid_std_after_normalization"] - 1.0)
                        .abs()
                        .max()
                    ),
                    "all_joint_means_within_tolerance": bool(
                        (
                            dataframe["joint_valid_mean_after_normalization"].abs()
                            <= VALIDATION_TOLERANCE
                        ).all()
                    ),
                    "all_joint_stds_within_tolerance": bool(
                        (
                            (
                                dataframe["joint_valid_std_after_normalization"]
                                - 1.0
                            ).abs()
                            <= VALIDATION_TOLERANCE
                        ).all()
                    ),
                    "normalization_std_min": float(dataframe["normalization_std"].min()),
                    "normalization_std_median": float(
                        dataframe["normalization_std"].median()
                    ),
                    "normalization_std_max": float(dataframe["normalization_std"].max()),
                }
            ),
            "final_output_validation": pd.Series(
                {
                    "correct_shape_count": int(dataframe["correct_shape"].sum()),
                    "float32_count": int(dataframe["float32_dtype"].sum()),
                    "contiguous_count": int(dataframe["c_contiguous"].sum()),
                    "nan_failures": int(dataframe["has_nan"].sum()),
                    "posinf_failures": int(dataframe["has_posinf"].sum()),
                    "neginf_failures": int(dataframe["has_neginf"].sum()),
                    "xy_padding_failures": int(
                        (
                            (dataframe["xy_padding_x_total"] > 0)
                            | (dataframe["xy_padding_y_total"] > 0)
                        ).sum()
                    ),
                    "z_cropping_failures": int((dataframe["z_cropping_total"] > 0).sum()),
                    "std_epsilon_failures": int(
                        (~dataframe["normalization_std_gt_epsilon"]).sum()
                    ),
                }
            ),
        }

    def _generate_qa_package(self, dataframe: pd.DataFrame) -> list[dict[str, str]]:
        """Generate selected patient QA figures and one contact sheet."""
        patient_ids = self._select_qa_patients(dataframe)
        entries: list[dict[str, str]] = []
        image_paths: list[Path] = []
        for patient_id in patient_ids:
            path = self.qa_dir / f"{patient_id}_final_preprocessing_qa.png"
            self._generate_patient_qa(patient_id, path)
            image_paths.append(path)
            entries.append({"patient_id": patient_id, "file": path.name})
        self._make_contact_sheet(image_paths)
        self._write_manifest(entries)
        return entries

    def _select_qa_patients(self, dataframe: pd.DataFrame) -> list[str]:
        """Select QA patients from padding extremes, spacing extremes, and classes."""
        selected: list[str] = []
        self._append_unique(
            selected,
            dataframe.nlargest(5, "z_padding_fraction")["patient_id"].tolist(),
        )
        self._append_unique(
            selected,
            dataframe.nsmallest(5, "z_padding_fraction")["patient_id"].tolist(),
        )
        xy_spacing = dataframe.assign(
            mean_xy_spacing=(
                dataframe["source_spacing_x_mm"] + dataframe["source_spacing_y_mm"]
            )
            / 2
        )
        self._append_unique(
            selected,
            [
                xy_spacing.sort_values("mean_xy_spacing").iloc[0]["patient_id"],
                xy_spacing.sort_values("mean_xy_spacing", ascending=False).iloc[0][
                    "patient_id"
                ],
                dataframe.sort_values("source_spacing_z_mm").iloc[0]["patient_id"],
                dataframe.sort_values("source_spacing_z_mm", ascending=False).iloc[0][
                    "patient_id"
                ],
            ],
        )
        for class_name in sorted(dataframe["class"].unique()):
            class_rows = dataframe[dataframe["class"] == class_name].copy()
            median_padding = class_rows["z_padding_fraction"].median()
            class_rows["distance_to_median_padding"] = (
                class_rows["z_padding_fraction"] - median_padding
            ).abs()
            self._append_unique(
                selected,
                [class_rows.sort_values("distance_to_median_padding").iloc[0]["patient_id"]],
            )
        return selected

    def _generate_patient_qa(self, patient_id: str, output_path: Path) -> None:
        """Generate one six-panel QA figure from actual processed arrays."""
        patient_dir = self.dataset_dir / patient_id
        record = self._process_patient(patient_dir, keep_tensor=True)
        ed_source = record["_ed_source"]
        es_source = record["_es_source"]
        ed_pre = record["_ed_pre_normalization"]
        es_pre = record["_es_pre_normalization"]
        tensor = record["_tensor"]
        source_mid_z = ed_source.shape[2] // 2
        pre_mid_z = ed_pre.shape[2] // 2
        final_mid_z = TARGET_DEPTH // 2

        figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        panels = [
            ("raw ED mid-Z", ed_source[:, :, source_mid_z]),
            ("raw ES mid-Z", es_source[:, :, source_mid_z]),
            ("resampled/cropped ED before normalization", ed_pre[:, :, pre_mid_z]),
            ("resampled/cropped ES before normalization", es_pre[:, :, pre_mid_z]),
            ("final normalized ED representative slice", tensor[0, final_mid_z]),
            ("final normalized ES representative slice", tensor[1, final_mid_z]),
        ]
        for axis, (title, image) in zip(axes.ravel(), panels, strict=True):
            axis.imshow(np.rot90(image), cmap="gray")
            axis.set_title(title)
            axis.set_axis_off()
        figure.suptitle(
            f"{patient_id} class={record['class']} "
            f"spacing=({record['source_spacing_x_mm']:.2f}, "
            f"{record['source_spacing_y_mm']:.2f}, "
            f"{record['source_spacing_z_mm']:.2f}) mm "
            f"resampled_depth={record['resampled_size_z']} "
            f"z_padding={record['z_padding_lower']}+{record['z_padding_upper']}"
        )
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    def _make_contact_sheet(self, image_paths: list[Path]) -> None:
        """Create high-resolution contact sheet for final preprocessing QA."""
        images = [Image.open(path).convert("RGB") for path in image_paths]
        thumb_width = 1200
        label_height = 52
        padding = 24
        columns = 2
        thumbs = []
        for image in images:
            scale = thumb_width / image.width
            thumb_height = int(round(image.height * scale))
            thumbs.append(image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS))
        cell_height = max(image.height for image in thumbs) + label_height
        rows = math.ceil(len(thumbs) / columns)
        sheet = Image.new(
            "RGB",
            (
                columns * thumb_width + (columns + 1) * padding,
                rows * cell_height + (rows + 1) * padding,
            ),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                28,
            )
        except OSError:
            font = ImageFont.load_default()
        for index, (path, image) in enumerate(zip(image_paths, thumbs, strict=True)):
            row = index // columns
            column = index % columns
            x0 = padding + column * (thumb_width + padding)
            y0 = padding + row * (cell_height + padding)
            draw.text((x0 + 8, y0 + 8), path.stem, fill="black", font=font)
            sheet.paste(image, (x0, y0 + label_height))
        sheet.save(self.qa_contact_sheet)

    def _write_manifest(self, entries: list[dict[str, str]]) -> None:
        """Write final preprocessing QA manifest in contact-sheet order."""
        lines = [f"{entry['patient_id']}\t{entry['file']}" for entry in entries]
        self.qa_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_report(
        self,
        summary: dict[str, pd.Series | pd.DataFrame],
        qa_entries: list[dict[str, str]],
    ) -> None:
        """Write final preprocessing validation markdown report."""
        report = [
            "# Final Preprocessing Validation",
            "",
            "## Frozen Spatial Policy",
            "- Authoritative spatial images: standalone ED and ES NIfTI volumes",
            "- Target orientation: LPS",
            "- Localization: deterministic geometric FOV-center",
            "- Target spacing: 1.50 x 1.50 x 7.50 mm",
            "- Final tensor shape: [2, 14, 144, 144]",
            "- Final center-to-center Z span: (14 - 1) * 7.5 = 97.5 mm",
            "",
            "## Normalization Formula",
            "Joint ED/ES per-patient clipping uses p0.5 and p99.5 over valid "
            "resampled/cropped voxels before artificial Z padding. The clipped "
            "joint ED+ES values define one mean and standard deviation, then both "
            "phases use `(clipped - mean) / max(std, 1e-6)`.",
            "",
            "## Array Transpose",
            "Each processed phase remains [X, Y, Z] until final tensor construction. "
            "Each phase is transposed with `transpose(2, 1, 0)` to [D, H, W], then "
            "ED and ES are stacked as channels 0 and 1.",
            "",
            "## Cohort Validation Summary",
            "### Resampled Depth",
            summary["resampled_depth"].to_markdown(),
            "",
            "### Z Padding",
            summary["z_padding"].to_markdown(),
            "",
            "### Z Padding By Class",
            summary["z_padding_by_class"].to_markdown(),
            "",
            "### Clipping Bounds",
            summary["clipping_bounds"].to_markdown(),
            "",
            "### Normalization Validation",
            summary["normalization_validation"].to_markdown(),
            "",
            "### Final Output Validation",
            summary["final_output_validation"].to_markdown(),
            "",
            "## Residual Padding Shortcut Diagnostic",
            f"- OOF Accuracy: {self.residual_shortcut_metrics['accuracy']:.3f}",
            f"- OOF Macro F1: {self.residual_shortcut_metrics['macro_f1']:.3f}",
            "- Chance accuracy: 0.200",
            "",
            "## QA Paths",
            f"- Contact sheet: `{self.qa_contact_sheet}`",
            f"- Manifest: `{self.qa_manifest}`",
            f"- QA figures: {len(qa_entries)}",
            "",
            "## Validation Failures",
            "None." if not self.validation_failures else "\n".join(self.validation_failures),
        ]
        self.output_report.write_text("\n".join(report) + "\n", encoding="utf-8")

    def _target_size_from_center_span(
        self,
        source_shape: tuple[int, int, int],
        source_spacing: tuple[float, float, float],
    ) -> tuple[int, int, int]:
        """Reuse center-span convention: round(span / target_spacing) + 1."""
        return tuple(
            max(
                1,
                int(round(((source_shape[axis] - 1) * source_spacing[axis]) / TARGET_SPACING_MM[axis]))
                + 1,
            )
            for axis in range(3)
        )

    def _six_number_summary(self, series: pd.Series) -> pd.Series:
        """Return min/p05/median/mean/p95/max summary."""
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

    def _parse_info_file(self, info_path: Path) -> dict[str, str]:
        """Read key-value metadata from one ACDC Info.cfg file."""
        metadata: dict[str, str] = {}
        with info_path.open("r", encoding="utf-8") as file:
            for line in file:
                key, value = line.strip().split(":", maxsplit=1)
                metadata[key.strip()] = value.strip()
        return metadata

    def _spatial_spacing(
        self,
        image: nib.spatialimages.SpatialImage,
    ) -> tuple[float, float, float]:
        """Read image spatial voxel spacing."""
        return tuple(float(value) for value in image.header.get_zooms()[:3])

    def _load_class_order(self) -> list[str]:
        """Load authoritative class ordering from config."""
        with self.class_mapping_path.open("r", encoding="utf-8") as file:
            mapping: dict[str, str] = json.load(file)
        return [mapping[key] for key in sorted(mapping, key=lambda value: int(value))]

    def _append_unique(self, selected: list[str], patient_ids: list[Any]) -> None:
        """Append patient IDs while preserving order and uniqueness."""
        seen = set(selected)
        for patient_id in patient_ids:
            patient_id = str(patient_id)
            if patient_id not in seen:
                selected.append(patient_id)
                seen.add(patient_id)

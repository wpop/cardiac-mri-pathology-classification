"""Integrated Phase 1 spatial-policy candidate analysis."""

import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cardiac_pathology_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


XY_TARGET_SPACINGS_MM = [1.25, 1.50, 1.75]
XY_REQUESTED_CROPS_MM = [180.0, 200.0, 220.0, 240.0]
Z_COMMON_SPACINGS_MM = [5.0, 7.5, 10.0]
Z_TARGET_CENTER_SPANS_MM = [70.0, 80.0, 90.0, 100.0, 110.0]
Z_NATIVE_TARGET_DEPTHS = [8, 10, 12, 14, 16]


class SpatialPolicyAnalyzer:
    """Evaluate remaining deterministic spatial preprocessing candidates."""

    def __init__(
        self,
        dataset_dir: Path,
        geometry_csv: Path,
        class_mapping_path: Path,
        output_dir: Path,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.geometry_csv = geometry_csv
        self.class_mapping_path = class_mapping_path
        self.output_dir = output_dir
        self.xy_candidates_csv = output_dir / "spatial_xy_candidates.csv"
        self.z_candidates_csv = output_dir / "spatial_z_candidates.csv"
        self.patient_simulation_csv = output_dir / "spatial_patient_simulation.csv"
        self.qa_contact_sheet = output_dir / "spatial_policy_qa_contact_sheet.png"
        self.qa_manifest = output_dir / "spatial_policy_qa_manifest.txt"
        self.summary_report = output_dir / "spatial_policy_summary.md"

    def run(self) -> dict[str, Any]:
        """Run candidate simulations, diagnostics, QA generation, and reporting."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        geometry = pd.read_csv(self.geometry_csv)
        class_order = self._load_class_order()

        xy_patient = self._simulate_xy_candidates(geometry)
        xy_summary = self._summarize_xy_candidates(xy_patient, geometry, class_order)
        z_patient = self._simulate_z_candidates(geometry)
        z_summary = self._summarize_z_candidates(z_patient, geometry, class_order)
        xy_shortlist = self._shortlist_xy(xy_summary)
        z_shortlist = self._shortlist_z(z_summary)

        patient_simulation = pd.concat([xy_patient, z_patient], ignore_index=True)
        xy_summary.to_csv(self.xy_candidates_csv, index=False)
        z_summary.to_csv(self.z_candidates_csv, index=False)
        patient_simulation.to_csv(self.patient_simulation_csv, index=False)

        qa_entries = self._generate_qa_contact_sheet(geometry, xy_patient, z_patient, z_shortlist)
        self._write_summary_report(xy_summary, z_summary, xy_shortlist, z_shortlist)

        return {
            "xy_summary": xy_summary,
            "z_summary": z_summary,
            "xy_shortlist": xy_shortlist,
            "z_shortlist": z_shortlist,
            "qa_entries": qa_entries,
            "qa_contact_sheet": self.qa_contact_sheet,
            "qa_manifest": self.qa_manifest,
            "summary_report": self.summary_report,
            "xy_candidates_csv": self.xy_candidates_csv,
            "z_candidates_csv": self.z_candidates_csv,
            "patient_simulation_csv": self.patient_simulation_csv,
        }

    def print_final_summaries(self, results: dict[str, Any]) -> None:
        """Print the required concise final summaries."""
        print("Spatial policy analysis")
        print("\nComplete XY candidate table")
        print(results["xy_summary"].round(4).to_string(index=False))
        print("\nBest 3 practical XY candidates")
        print(results["xy_shortlist"].round(4).to_string(index=False))
        print("\nComplete Z candidate table")
        print(results["z_summary"].round(4).to_string(index=False))
        print("\nBest 3 practical Z candidates")
        print(results["z_shortlist"].round(4).to_string(index=False))

        with Image.open(results["qa_contact_sheet"]) as image:
            dimensions = f"{image.width}x{image.height}"
        print("\nQA")
        print(f"contact_sheet: {results['qa_contact_sheet']}")
        print(f"contact_sheet_dimensions: {dimensions}")
        print("\nGenerated paths")
        print(f"xy_candidates_csv: {results['xy_candidates_csv']}")
        print(f"z_candidates_csv: {results['z_candidates_csv']}")
        print(f"patient_simulation_csv: {results['patient_simulation_csv']}")
        print(f"summary_report: {results['summary_report']}")
        print(f"qa_manifest: {results['qa_manifest']}")

    def _simulate_xy_candidates(self, geometry: pd.DataFrame) -> pd.DataFrame:
        """Simulate target XY spacing and centered square physical crops."""
        records: list[dict[str, Any]] = []
        for row in geometry.itertuples(index=False):
            for target_spacing in XY_TARGET_SPACINGS_MM:
                span_x_mm = (row.size_x - 1) * row.spacing_x_mm
                span_y_mm = (row.size_y - 1) * row.spacing_y_mm
                resampled_x = self._size_preserving_center_span(span_x_mm, target_spacing)
                resampled_y = self._size_preserving_center_span(span_y_mm, target_spacing)
                scale_x = row.spacing_x_mm / target_spacing
                scale_y = row.spacing_y_mm / target_spacing
                abs_log_scale_x = abs(math.log(scale_x))
                abs_log_scale_y = abs(math.log(scale_y))
                burden = max(abs_log_scale_x, abs_log_scale_y)

                for requested_crop_mm in XY_REQUESTED_CROPS_MM:
                    target_pixels = self._round_up_to_multiple(
                        math.ceil(requested_crop_mm / target_spacing),
                        16,
                    )
                    actual_crop_mm = target_pixels * target_spacing
                    x_operation = self._split_crop_pad(resampled_x, target_pixels)
                    y_operation = self._split_crop_pad(resampled_y, target_pixels)
                    valid_x = min(resampled_x, target_pixels)
                    valid_y = min(resampled_y, target_pixels)
                    valid_area = valid_x * valid_y
                    final_area = target_pixels * target_pixels
                    original_area = resampled_x * resampled_y

                    records.append(
                        {
                            "analysis_type": "xy",
                            "patient_id": row.patient_id,
                            "class": self._row_class(row),
                            "target_spacing_xy_mm": target_spacing,
                            "requested_crop_mm": requested_crop_mm,
                            "target_pixels": target_pixels,
                            "actual_crop_mm": actual_crop_mm,
                            "scale_x": scale_x,
                            "scale_y": scale_y,
                            "absolute_scale_change_x": abs(scale_x - 1.0),
                            "absolute_scale_change_y": abs(scale_y - 1.0),
                            "absolute_log_scale_change_x": abs_log_scale_x,
                            "absolute_log_scale_change_y": abs_log_scale_y,
                            "resampling_burden": burden,
                            "upsamples_x": scale_x > 1.0,
                            "upsamples_y": scale_y > 1.0,
                            "downsamples_x": scale_x < 1.0,
                            "downsamples_y": scale_y < 1.0,
                            "resampled_size_x": resampled_x,
                            "resampled_size_y": resampled_y,
                            "crop_x_lower_pixels": x_operation["crop_lower"],
                            "crop_x_upper_pixels": x_operation["crop_upper"],
                            "crop_y_lower_pixels": y_operation["crop_lower"],
                            "crop_y_upper_pixels": y_operation["crop_upper"],
                            "pad_x_lower_pixels": x_operation["pad_lower"],
                            "pad_x_upper_pixels": x_operation["pad_upper"],
                            "pad_y_lower_pixels": y_operation["pad_lower"],
                            "pad_y_upper_pixels": y_operation["pad_upper"],
                            "padding_x_fraction": x_operation["pad_total"] / target_pixels,
                            "padding_y_fraction": y_operation["pad_total"] / target_pixels,
                            "cropped_x_fraction": x_operation["crop_total"] / resampled_x,
                            "cropped_y_fraction": y_operation["crop_total"] / resampled_y,
                            "padding_fraction_xy": 1.0 - valid_area / final_area,
                            "cropped_fraction_xy": 1.0 - valid_area / original_area,
                            "retained_original_fov_fraction": valid_area / original_area,
                            "physical_padding_x_mm": x_operation["pad_total"] * target_spacing,
                            "physical_padding_y_mm": y_operation["pad_total"] * target_spacing,
                            "physical_cropping_x_mm": x_operation["crop_total"] * target_spacing,
                            "physical_cropping_y_mm": y_operation["crop_total"] * target_spacing,
                        }
                    )

        return pd.DataFrame(records)

    def _simulate_z_candidates(self, geometry: pd.DataFrame) -> pd.DataFrame:
        """Simulate native-Z and common-Z spacing/depth candidates."""
        records: list[dict[str, Any]] = []
        for row in geometry.itertuples(index=False):
            for target_depth in Z_NATIVE_TARGET_DEPTHS:
                records.append(
                    self._simulate_one_z_candidate(
                        row=row,
                        policy="native",
                        target_spacing_z_mm=np.nan,
                        target_center_span_mm=np.nan,
                        resulting_depth=row.size_z,
                        target_depth=target_depth,
                        resampling_burden=0.0,
                    )
                )

            old_center_span_mm = (row.size_z - 1) * row.spacing_z_mm
            for target_spacing_z in Z_COMMON_SPACINGS_MM:
                resulting_depth = self._size_preserving_center_span(
                    old_center_span_mm,
                    target_spacing_z,
                )
                resampling_burden = abs(math.log(row.spacing_z_mm / target_spacing_z))
                for target_center_span in Z_TARGET_CENTER_SPANS_MM:
                    target_depth = self._size_preserving_center_span(
                        target_center_span,
                        target_spacing_z,
                    )
                    records.append(
                        self._simulate_one_z_candidate(
                            row=row,
                            policy="common_spacing",
                            target_spacing_z_mm=target_spacing_z,
                            target_center_span_mm=target_center_span,
                            resulting_depth=resulting_depth,
                            target_depth=target_depth,
                            resampling_burden=resampling_burden,
                        )
                    )

        return pd.DataFrame(records)

    def _simulate_one_z_candidate(
        self,
        row: Any,
        policy: str,
        target_spacing_z_mm: float,
        target_center_span_mm: float,
        resulting_depth: int,
        target_depth: int,
        resampling_burden: float,
    ) -> dict[str, Any]:
        """Simulate one Z candidate for one patient."""
        operation = self._split_crop_pad(resulting_depth, target_depth)
        valid_slices = min(resulting_depth, target_depth)
        effective_spacing = row.spacing_z_mm if policy == "native" else target_spacing_z_mm

        return {
            "analysis_type": "z",
            "patient_id": row.patient_id,
            "class": self._row_class(row),
            "policy": policy,
            "target_spacing_z_mm": target_spacing_z_mm,
            "target_center_span_mm": target_center_span_mm,
            "target_D": target_depth,
            "resulting_resampled_depth": resulting_depth,
            "required_padding_slices": operation["pad_total"],
            "required_crop_slices": operation["crop_total"],
            "padding_lower_slices": operation["pad_lower"],
            "padding_upper_slices": operation["pad_upper"],
            "crop_lower_slices": operation["crop_lower"],
            "crop_upper_slices": operation["crop_upper"],
            "padding_fraction": operation["pad_total"] / target_depth,
            "crop_fraction": operation["crop_total"] / resulting_depth,
            "padded_slice_fraction": operation["pad_total"] / target_depth,
            "cropped_slice_fraction": operation["crop_total"] / resulting_depth,
            "valid_slice_fraction": valid_slices / target_depth,
            "physical_amount_padded_mm": operation["pad_total"] * effective_spacing,
            "physical_amount_cropped_mm": operation["crop_total"] * effective_spacing,
            "any_z_padding": operation["pad_total"] > 0,
            "any_z_crop": operation["crop_total"] > 0,
            "resampling_burden": resampling_burden,
        }

    def _summarize_xy_candidates(
        self,
        xy_patient: pd.DataFrame,
        geometry: pd.DataFrame,
        class_order: list[str],
    ) -> pd.DataFrame:
        """Build one-row-per-XY-candidate summary with shortcut diagnostics."""
        summaries: list[dict[str, Any]] = []
        group_columns = ["target_spacing_xy_mm", "requested_crop_mm"]
        for (spacing, crop), group in xy_patient.groupby(group_columns, sort=True):
            metrics = self._run_shortcut_baseline(
                group=group,
                geometry=geometry,
                feature_columns=[
                    "padding_fraction_xy",
                    "padding_x_fraction",
                    "padding_y_fraction",
                    "cropped_x_fraction",
                    "cropped_y_fraction",
                    "resampled_size_x",
                    "resampled_size_y",
                ],
                class_order=class_order,
            )
            summaries.append(
                {
                    "target_spacing_xy_mm": spacing,
                    "requested_crop_mm": crop,
                    "actual_crop_mm": float(group["actual_crop_mm"].iloc[0]),
                    "target_pixels": int(group["target_pixels"].iloc[0]),
                    "median_resampling_burden": group["resampling_burden"].median(),
                    "p95_resampling_burden": group["resampling_burden"].quantile(0.95),
                    "patients_with_padding": int((group["padding_fraction_xy"] > 0).sum()),
                    "median_padding_fraction": group["padding_fraction_xy"].median(),
                    "p95_padding_fraction": group["padding_fraction_xy"].quantile(0.95),
                    "max_padding_fraction": group["padding_fraction_xy"].max(),
                    "median_cropped_fraction": group["cropped_fraction_xy"].median(),
                    "p95_cropped_fraction": group["cropped_fraction_xy"].quantile(0.95),
                    "OOF_accuracy": metrics["accuracy"],
                    "OOF_macro_f1": metrics["macro_f1"],
                }
            )

        return pd.DataFrame(summaries)

    def _summarize_z_candidates(
        self,
        z_patient: pd.DataFrame,
        geometry: pd.DataFrame,
        class_order: list[str],
    ) -> pd.DataFrame:
        """Build one-row-per-Z-candidate summary with shortcut diagnostics."""
        summaries: list[dict[str, Any]] = []
        group_columns = ["policy", "target_spacing_z_mm", "target_center_span_mm", "target_D"]
        z_patient_groupable = z_patient.copy()
        z_patient_groupable["target_spacing_z_mm"] = z_patient_groupable[
            "target_spacing_z_mm"
        ].fillna("native")
        z_patient_groupable["target_center_span_mm"] = z_patient_groupable[
            "target_center_span_mm"
        ].fillna("native")

        for key, group in z_patient_groupable.groupby(group_columns, sort=True):
            policy, spacing, span, target_depth = key
            metrics = self._run_shortcut_baseline(
                group=group,
                geometry=geometry,
                feature_columns=[
                    "padding_fraction",
                    "crop_fraction",
                    "padded_slice_fraction",
                    "cropped_slice_fraction",
                    "valid_slice_fraction",
                    "required_padding_slices",
                    "required_crop_slices",
                ],
                class_order=class_order,
            )
            summaries.append(
                {
                    "policy": policy,
                    "target_spacing_z_mm": spacing,
                    "target_center_span_mm": span,
                    "target_D": int(target_depth),
                    "patients_with_padding": int(group["any_z_padding"].sum()),
                    "patients_with_cropping": int(group["any_z_crop"].sum()),
                    "median_padding_fraction": group["padding_fraction"].median(),
                    "p95_padding_fraction": group["padding_fraction"].quantile(0.95),
                    "max_padding_fraction": group["padding_fraction"].max(),
                    "median_cropped_fraction": group["crop_fraction"].median(),
                    "p95_cropped_fraction": group["crop_fraction"].quantile(0.95),
                    "max_cropped_fraction": group["crop_fraction"].max(),
                    "median_resampling_burden": group["resampling_burden"].median(),
                    "OOF_accuracy": metrics["accuracy"],
                    "OOF_macro_f1": metrics["macro_f1"],
                }
            )

        return pd.DataFrame(summaries)

    def _run_shortcut_baseline(
        self,
        group: pd.DataFrame,
        geometry: pd.DataFrame,
        feature_columns: list[str],
        class_order: list[str],
    ) -> dict[str, float]:
        """Run leakage-safe OOF diagnostic baseline on simulated artifacts."""
        merged = (
            group[["patient_id", *feature_columns]]
            .merge(geometry[["patient_id", "class"]], on="patient_id", how="left")
            .sort_values("patient_id")
        )
        features = merged[feature_columns]
        labels = merged["class"]
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

    def _shortlist_xy(self, xy_summary: pd.DataFrame) -> pd.DataFrame:
        """Return fixed-criteria top three practical XY candidates."""
        shortlist = xy_summary.copy()
        shortlist["meets_padding_count_rule"] = shortlist["patients_with_padding"] <= 5
        shortlist["meets_p95_padding_rule"] = shortlist["p95_padding_fraction"] <= 0.02
        shortlist["practical_rule_count"] = (
            shortlist["meets_padding_count_rule"].astype(int)
            + shortlist["meets_p95_padding_rule"].astype(int)
        )
        return (
            shortlist.sort_values(
                [
                    "practical_rule_count",
                    "median_resampling_burden",
                    "OOF_macro_f1",
                    "p95_padding_fraction",
                    "p95_cropped_fraction",
                ],
                ascending=[False, True, True, True, True],
            )
            .head(3)
            .drop(columns=["meets_padding_count_rule", "meets_p95_padding_rule"])
        )

    def _shortlist_z(self, z_summary: pd.DataFrame) -> pd.DataFrame:
        """Return fixed-criteria top three practical Z candidates."""
        return (
            z_summary.sort_values(
                [
                    "p95_cropped_fraction",
                    "patients_with_cropping",
                    "OOF_macro_f1",
                    "median_resampling_burden",
                    "p95_padding_fraction",
                ],
                ascending=[True, True, True, True, True],
            )
            .head(3)
        )

    def _generate_qa_contact_sheet(
        self,
        geometry: pd.DataFrame,
        xy_patient: pd.DataFrame,
        z_patient: pd.DataFrame,
        z_shortlist: pd.DataFrame,
    ) -> list[dict[str, str]]:
        """Generate one contact sheet covering XY and Z QA selections."""
        qa_dir = self.output_dir / "spatial_policy_qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, str]] = []
        image_paths: list[Path] = []

        xy_patients = self._select_xy_qa_patients(geometry, xy_patient)
        for patient_id in xy_patients:
            path = qa_dir / f"{patient_id}_xy_crop_qa.png"
            self._generate_xy_qa(patient_id, path)
            image_paths.append(path)
            entries.append({"section": "XY", "patient_id": patient_id, "file": path.name})

        z_patients = self._select_z_qa_patients(geometry, z_patient, z_shortlist)
        for _, candidate in z_shortlist.iterrows():
            candidate_id = self._z_candidate_id(candidate)
            for patient_id in z_patients:
                path = qa_dir / f"{patient_id}_{candidate_id}_z_qa.png"
                self._generate_z_qa(patient_id, candidate, z_patient, path)
                image_paths.append(path)
                entries.append(
                    {
                        "section": f"Z {candidate_id}",
                        "patient_id": patient_id,
                        "file": path.name,
                    }
                )

        self._make_contact_sheet(image_paths)
        self._write_manifest(entries)
        return entries

    def _select_xy_qa_patients(
        self,
        geometry: pd.DataFrame,
        xy_patient: pd.DataFrame,
    ) -> list[str]:
        """Select patients for XY visual crop QA."""
        selected: list[str] = []
        geometry = geometry.copy()
        geometry["fov_area_mm2"] = geometry["fov_x_mm"] * geometry["fov_y_mm"]
        self._append_unique(selected, [geometry.sort_values("fov_area_mm2").iloc[0]["patient_id"]])

        xy_150 = xy_patient[xy_patient["target_spacing_xy_mm"] == 1.50]
        burden = xy_150.groupby("patient_id")["resampling_burden"].max()
        self._append_unique(selected, [burden.sort_values(ascending=False).index[0]])

        motion_csv = self.output_dir / "motion_localization.csv"
        if motion_csv.is_file():
            motion = pd.read_csv(motion_csv)
            worst_motion = motion.nlargest(5, "p975_centroid_agreement_mm")
            self._append_unique(selected, worst_motion["patient_id"].tolist())

        for class_name in sorted(geometry["class"].unique()):
            class_geometry = geometry[geometry["class"] == class_name].copy()
            median_area = class_geometry["fov_area_mm2"].median()
            class_geometry["distance_to_median_area"] = (
                class_geometry["fov_area_mm2"] - median_area
            ).abs()
            self._append_unique(
                selected,
                [class_geometry.sort_values("distance_to_median_area").iloc[0]["patient_id"]],
            )

        return selected

    def _select_z_qa_patients(
        self,
        geometry: pd.DataFrame,
        z_patient: pd.DataFrame,
        z_shortlist: pd.DataFrame,
    ) -> list[str]:
        """Select patients for Z range QA using best-three Z candidates."""
        selected: list[str] = []
        best_groups = []
        for _, candidate in z_shortlist.iterrows():
            best_groups.append(self._z_candidate_rows(z_patient, candidate))
        best_patient = pd.concat(best_groups, ignore_index=True)
        crop_by_patient = best_patient.groupby("patient_id")["required_crop_slices"].max()
        pad_by_patient = best_patient.groupby("patient_id")["required_padding_slices"].max()
        self._append_unique(selected, [crop_by_patient.sort_values(ascending=False).index[0]])
        self._append_unique(selected, [pad_by_patient.sort_values(ascending=False).index[0]])
        self._append_unique(selected, [geometry.sort_values("z_center_span_mm").iloc[0]["patient_id"]])
        self._append_unique(
            selected,
            [geometry.sort_values("z_center_span_mm", ascending=False).iloc[0]["patient_id"]],
        )

        for class_name in sorted(geometry["class"].unique()):
            class_geometry = geometry[geometry["class"] == class_name].copy()
            median_depth = class_geometry["size_z"].median()
            class_geometry["distance_to_median_depth"] = (
                class_geometry["size_z"] - median_depth
            ).abs()
            self._append_unique(
                selected,
                [class_geometry.sort_values("distance_to_median_depth").iloc[0]["patient_id"]],
            )

        return selected

    def _generate_xy_qa(self, patient_id: str, output_path: Path) -> None:
        """Generate one XY crop-comparison QA image from raw ED/ES mid-Z slices."""
        patient_dir = self.dataset_dir / patient_id
        metadata = self._parse_info_file(patient_dir / "Info.cfg")
        ed_frame = int(metadata["ED"])
        es_frame = int(metadata["ES"])
        ed_image = nib.load(patient_dir / f"{patient_id}_frame{ed_frame:02d}.nii.gz")
        es_image = nib.load(patient_dir / f"{patient_id}_frame{es_frame:02d}.nii.gz")
        ed = np.asanyarray(ed_image.dataobj)
        es = np.asanyarray(es_image.dataobj)
        spacing_x, spacing_y, _spacing_z = self._spatial_spacing(ed_image)
        z_index = ed.shape[2] // 2

        figure, axes = plt.subplots(4, 4, figsize=(14, 14), constrained_layout=True)
        for row_index, crop_mm in enumerate(XY_REQUESTED_CROPS_MM):
            self._plot_full_with_crop(
                axes[row_index, 0],
                ed[:, :, z_index],
                spacing_x,
                spacing_y,
                crop_mm,
                f"{patient_id} ED full {crop_mm:.0f} mm",
            )
            self._plot_full_with_crop(
                axes[row_index, 1],
                es[:, :, z_index],
                spacing_x,
                spacing_y,
                crop_mm,
                f"{patient_id} ES full {crop_mm:.0f} mm",
            )
            axes[row_index, 2].imshow(
                np.rot90(self._center_crop_raw(ed[:, :, z_index], spacing_x, spacing_y, crop_mm)),
                cmap="gray",
            )
            axes[row_index, 2].set_title(f"ED centered crop {crop_mm:.0f} mm")
            axes[row_index, 3].imshow(
                np.rot90(self._center_crop_raw(es[:, :, z_index], spacing_x, spacing_y, crop_mm)),
                cmap="gray",
            )
            axes[row_index, 3].set_title(f"ES centered crop {crop_mm:.0f} mm")
            for column in range(4):
                axes[row_index, column].set_axis_off()

        figure.suptitle(f"{patient_id} spatial policy XY QA at 1.50 mm target spacing")
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    def _generate_z_qa(
        self,
        patient_id: str,
        candidate: pd.Series,
        z_patient: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Generate one Z retained/cropped/padded range QA image."""
        row = self._z_candidate_rows(z_patient, candidate)
        patient_row = row[row["patient_id"] == patient_id].iloc[0]
        geometry = pd.read_csv(self.geometry_csv)
        geom = geometry[geometry["patient_id"] == patient_id].iloc[0]
        target_depth = int(patient_row["target_D"])
        resulting_depth = int(patient_row["resulting_resampled_depth"])
        crop_lower = int(patient_row["crop_lower_slices"])
        crop_upper = int(patient_row["crop_upper_slices"])
        pad_lower = int(patient_row["padding_lower_slices"])
        pad_upper = int(patient_row["padding_upper_slices"])
        retained_start = crop_lower
        retained_end = resulting_depth - crop_upper - 1

        figure, axis = plt.subplots(1, 1, figsize=(12, 3), constrained_layout=True)
        axis.set_title(
            f"{patient_id} Z QA {self._z_candidate_id(candidate)} "
            f"native_span={geom.z_center_span_mm:.1f} mm"
        )
        axis.broken_barh([(0, resulting_depth)], (10, 8), facecolors="lightgray")
        if crop_lower > 0:
            axis.broken_barh([(0, crop_lower)], (10, 8), facecolors="tab:red")
        if crop_upper > 0:
            axis.broken_barh(
                [(resulting_depth - crop_upper, crop_upper)],
                (10, 8),
                facecolors="tab:red",
            )
        if retained_end >= retained_start:
            axis.broken_barh(
                [(retained_start, retained_end - retained_start + 1)],
                (10, 8),
                facecolors="tab:green",
            )
        axis.broken_barh([(0, target_depth)], (24, 8), facecolors="lightgray")
        if pad_lower > 0:
            axis.broken_barh([(0, pad_lower)], (24, 8), facecolors="tab:blue")
        valid_start = pad_lower
        valid_count = min(resulting_depth, target_depth)
        axis.broken_barh([(valid_start, valid_count)], (24, 8), facecolors="tab:green")
        if pad_upper > 0:
            axis.broken_barh(
                [(target_depth - pad_upper, pad_upper)],
                (24, 8),
                facecolors="tab:blue",
            )
        axis.set_yticks([14, 28])
        axis.set_yticklabels(["simulated source", "final D"])
        axis.set_xlabel("Slice index")
        axis.set_xlim(0, max(resulting_depth, target_depth) + 1)
        axis.grid(axis="x", alpha=0.3)
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    def _plot_full_with_crop(
        self,
        axis: plt.Axes,
        image: np.ndarray,
        spacing_x: float,
        spacing_y: float,
        crop_mm: float,
        title: str,
    ) -> None:
        """Plot a full slice with centered physical crop rectangle."""
        axis.imshow(np.rot90(image), cmap="gray")
        width_vox = crop_mm / spacing_x
        height_vox = crop_mm / spacing_y
        center_x = (image.shape[0] - 1) / 2
        center_y = (image.shape[1] - 1) / 2
        x0 = center_x - width_vox / 2
        y0 = image.shape[1] - 1 - (center_y + height_vox / 2)
        axis.add_patch(
            Rectangle(
                (x0, y0),
                width_vox,
                height_vox,
                fill=False,
                edgecolor="cyan",
                linewidth=1.5,
            )
        )
        axis.set_title(title)

    def _center_crop_raw(
        self,
        image: np.ndarray,
        spacing_x: float,
        spacing_y: float,
        crop_mm: float,
    ) -> np.ndarray:
        """Extract a centered raw-grid crop corresponding to a physical crop."""
        crop_x = min(image.shape[0], max(1, int(round(crop_mm / spacing_x))))
        crop_y = min(image.shape[1], max(1, int(round(crop_mm / spacing_y))))
        start_x = (image.shape[0] - crop_x) // 2
        start_y = (image.shape[1] - crop_y) // 2
        return image[start_x : start_x + crop_x, start_y : start_y + crop_y]

    def _make_contact_sheet(self, image_paths: list[Path]) -> None:
        """Create a high-resolution readable contact sheet from QA PNGs."""
        images = [Image.open(path).convert("RGB") for path in image_paths]
        thumb_width = 1200
        label_height = 52
        padding = 24
        columns = 2
        thumbs: list[Image.Image] = []
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
        """Write contact-sheet manifest in display order."""
        lines = [
            f"{entry['section']}\t{entry['patient_id']}\t{entry['file']}"
            for entry in entries
        ]
        self.qa_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_summary_report(
        self,
        xy_summary: pd.DataFrame,
        z_summary: pd.DataFrame,
        xy_shortlist: pd.DataFrame,
        z_shortlist: pd.DataFrame,
    ) -> None:
        """Write the Phase 1 spatial-policy summary report."""
        frozen_facts = [
            "100 patients with exactly 20 patients per class.",
            "Standalone ED/ES files exist for all patients.",
            "Standalone ED/ES are the authoritative spatial images.",
            "Standalone ED/ES orientation is LPS for all 100 patients.",
            "Raw 4D cine ED/ES frames exactly match standalone ED/ES arrays.",
            "Acquisition geometry contains measurable diagnosis-related shortcut information.",
            "Geometry-only Z baseline achieved OOF Accuracy 0.390.",
            "Exact zero, global minimum, and boundary intensity are not reliable background definitions.",
            "Temporal-variance and ED-ES connected-component localization are rejected.",
            "Deterministic geometric FOV-center localization is frozen for further Phase 1 work.",
        ]
        report = [
            "# Spatial Policy Summary",
            "",
            "## Frozen Facts Used As Inputs",
            *[f"- {fact}" for fact in frozen_facts],
            "",
            "## XY Candidate Table",
            xy_summary.round(4).to_markdown(index=False),
            "",
            "## Z Candidate Table",
            z_summary.round(4).to_markdown(index=False),
            "",
            "## Quantitative XY Shortlist",
            xy_shortlist.round(4).to_markdown(index=False),
            "",
            "## Quantitative Z Shortlist",
            z_shortlist.round(4).to_markdown(index=False),
            "",
            "## Decision Status",
            "No final spatial policy decision is made in this report.",
            "",
            "## QA Files",
            f"- Contact sheet: `{self.qa_contact_sheet}`",
            f"- Manifest: `{self.qa_manifest}`",
        ]
        self.summary_report.write_text("\n".join(report) + "\n", encoding="utf-8")

    def _z_candidate_rows(self, z_patient: pd.DataFrame, candidate: pd.Series) -> pd.DataFrame:
        """Return patient rows matching one Z candidate summary row."""
        spacing = candidate["target_spacing_z_mm"]
        span = candidate["target_center_span_mm"]
        rows = z_patient[
            (z_patient["policy"] == candidate["policy"])
            & (z_patient["target_D"] == candidate["target_D"])
        ]
        if candidate["policy"] == "native":
            return rows[rows["target_spacing_z_mm"].isna()]
        return rows[
            (rows["target_spacing_z_mm"] == float(spacing))
            & (rows["target_center_span_mm"] == float(span))
        ]

    def _z_candidate_id(self, candidate: pd.Series) -> str:
        """Create a compact file-safe Z candidate identifier."""
        if candidate["policy"] == "native":
            return f"z_native_D{int(candidate['target_D'])}"
        spacing = str(candidate["target_spacing_z_mm"]).replace(".", "p")
        span = str(candidate["target_center_span_mm"]).replace(".", "p")
        return f"z_s{spacing}_span{span}_D{int(candidate['target_D'])}"

    def _size_preserving_center_span(self, center_span_mm: float, spacing_mm: float) -> int:
        """Return round(center_span / spacing) + 1 to preserve center span closely."""
        return max(1, int(round(center_span_mm / spacing_mm)) + 1)

    def _round_up_to_multiple(self, value: int, multiple: int) -> int:
        """Round an integer up to the nearest multiple."""
        return int(math.ceil(value / multiple) * multiple)

    def _split_crop_pad(self, source_size: int, target_size: int) -> dict[str, int]:
        """Symmetrically split crop or padding, lower side gets floor(diff / 2)."""
        if source_size > target_size:
            difference = source_size - target_size
            lower = difference // 2
            upper = difference - lower
            return {
                "crop_lower": lower,
                "crop_upper": upper,
                "crop_total": difference,
                "pad_lower": 0,
                "pad_upper": 0,
                "pad_total": 0,
            }
        difference = target_size - source_size
        lower = difference // 2
        upper = difference - lower
        return {
            "crop_lower": 0,
            "crop_upper": 0,
            "crop_total": 0,
            "pad_lower": lower,
            "pad_upper": upper,
            "pad_total": difference,
        }

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
        """Load authoritative class ordering from the repository config."""
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

    def _row_class(self, row: Any) -> str:
        """Read the diagnostic class from a pandas itertuples row."""
        row_values = row._asdict()
        return str(row_values.get("class", row_values.get("_1")))

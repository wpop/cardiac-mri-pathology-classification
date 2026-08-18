"""Unit tests for the ONNX exporter contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from cardiac_pathology.deployment.onnx_exporter import OnnxExporter
from cardiac_pathology.models.resnet3d18 import ResNet3D18


def test_export_writes_valid_onnx_with_expected_io_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid Phase 9 checkpoint exports a classifier-only ONNX graph."""
    install_fake_onnx_export(monkeypatch)
    checkpoint_path = write_phase9_checkpoint(tmp_path / "classifier.pt")
    output_path = tmp_path / "deployment" / "classifier.onnx"
    output_path.parent.mkdir()

    exported_path = OnnxExporter().export(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
    )

    assert exported_path == output_path
    assert output_path.is_file()

    model = onnx.load(output_path)
    onnx.checker.check_model(model)

    graph_input = model.graph.input[0]
    graph_output = model.graph.output[0]
    input_type = graph_input.type.tensor_type
    output_type = graph_output.type.tensor_type

    assert graph_input.name == "cine_mri"
    assert graph_output.name == "logits"
    assert input_type.elem_type == TensorProto.FLOAT
    assert output_type.elem_type == TensorProto.FLOAT
    assert tensor_shape(input_type)[:1] == ["dynamic"]
    assert tensor_shape(input_type)[1:] == [2, 14, 144, 144]
    assert tensor_shape(output_type)[:1] == ["dynamic"]
    assert tensor_shape(output_type)[1:] == [5]


def test_export_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    """Exporter overwrite protection blocks accidental ONNX replacement."""
    checkpoint_path = write_phase9_checkpoint(tmp_path / "classifier.pt")
    output_path = tmp_path / "classifier.onnx"
    output_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        OnnxExporter().export(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
        )


@pytest.mark.parametrize(
    ("metadata_key", "invalid_value"),
    [
        ("architecture", "OtherNet"),
        ("input_channels", 1),
        ("num_classes", 4),
    ],
)
def test_export_rejects_invalid_checkpoint_metadata(
    tmp_path: Path,
    metadata_key: str,
    invalid_value: object,
) -> None:
    """Required Phase 9 checkpoint metadata must match the ONNX contract."""
    checkpoint_path = write_phase9_checkpoint(
        tmp_path / "classifier.pt",
        metadata_overrides={metadata_key: invalid_value},
    )

    with pytest.raises(ValueError, match=metadata_key):
        OnnxExporter().export(
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "classifier.onnx",
        )


def test_export_rejects_missing_model_state_dict(tmp_path: Path) -> None:
    """A checkpoint without model weights is rejected before export."""
    checkpoint_path = tmp_path / "classifier.pt"
    torch.save(
        {
            "architecture": "ResNet3D18",
            "input_channels": 2,
            "num_classes": 5,
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="model_state_dict"):
        OnnxExporter().export(
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "classifier.onnx",
        )


def write_phase9_checkpoint(
    checkpoint_path: Path,
    *,
    metadata_overrides: dict[str, object] | None = None,
) -> Path:
    """Write a temporary checkpoint with the required Phase 9 metadata."""
    model = ResNet3D18(input_channels=2, num_classes=5)
    payload: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "architecture": "ResNet3D18",
        "input_channels": 2,
        "num_classes": 5,
    }
    if metadata_overrides is not None:
        payload.update(metadata_overrides)

    torch.save(payload, checkpoint_path)
    return checkpoint_path


def tensor_shape(tensor_type: onnx.TypeProto.Tensor) -> list[int | str]:
    """Return ONNX tensor dimensions, marking symbolic dimensions as dynamic."""
    dimensions: list[int | str] = []
    for dimension in tensor_type.shape.dim:
        if dimension.dim_param:
            dimensions.append("dynamic")
        else:
            dimensions.append(dimension.dim_value)
    return dimensions


def install_fake_onnx_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a narrow ONNX writer test double for exporter unit tests."""

    def fake_onnx_export(
        model: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        f: str | Path,
        **kwargs: Any,
    ) -> None:
        assert isinstance(model, ResNet3D18)
        assert not model.training
        assert len(args) == 1
        assert tuple(args[0].shape) == (2, 2, 14, 144, 144)
        assert args[0].dtype == torch.float32
        assert kwargs["input_names"] == ["cine_mri"]
        assert kwargs["output_names"] == ["logits"]
        assert kwargs["opset_version"] == 18
        assert kwargs["dynamo"] is True
        assert kwargs["external_data"] is False
        assert kwargs["dynamic_shapes"][0][0].__name__ == "batch"

        onnx.save(
            make_minimal_classifier_onnx(),
            f,
        )

    monkeypatch.setattr(torch.onnx, "export", fake_onnx_export)


def make_minimal_classifier_onnx() -> onnx.ModelProto:
    """Create a minimal valid ONNX classifier graph for unit-test inspection."""
    cine_mri = helper.make_tensor_value_info(
        "cine_mri",
        TensorProto.FLOAT,
        ["batch", 2, 14, 144, 144],
    )
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 5])
    reduce_axes = numpy_helper.from_array(
        torch.tensor([1, 2, 3, 4], dtype=torch.int64).numpy(),
        name="reduce_axes",
    )
    unsqueeze_axes = numpy_helper.from_array(
        torch.tensor([1], dtype=torch.int64).numpy(),
        name="unsqueeze_axes",
    )
    tile_repeats = numpy_helper.from_array(
        torch.tensor([1, 5], dtype=torch.int64).numpy(),
        name="tile_repeats",
    )
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["cine_mri", "reduce_axes"], ["mean"], keepdims=0),
            helper.make_node("Unsqueeze", ["mean", "unsqueeze_axes"], ["mean_column"]),
            helper.make_node("Tile", ["mean_column", "tile_repeats"], ["logits"]),
        ],
        "minimal_classifier",
        [cine_mri],
        [logits],
        initializer=[reduce_axes, unsqueeze_axes, tile_repeats],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
    )

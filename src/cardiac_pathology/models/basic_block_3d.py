"""Reusable 3D ResNet basic residual block."""

from torch import Tensor, nn


class BasicBlock3D(nn.Module):
    """Basic residual block for a 3D ResNet-18-style architecture.

    The block processes 5D PyTorch tensors with shape ``[N, C, D, H, W]``, where
    ``N`` is batch size, ``C`` is channel count, and ``D``, ``H``, and ``W`` are
    depth, height, and width. It follows the standard ResNet basic-block pattern:
    two 3D convolutions with batch normalization, an optional residual projection,
    and a final ReLU after adding the residual path.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int, int] = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        """Initialize a 3D ResNet basic residual block.

        Args:
            in_channels: Number of channels in the input tensor.
            out_channels: Number of channels produced by each convolution in the
                main residual branch.
            stride: Stride for the first convolution. Use an integer for equal
                depth, height, and width stride, or a 3D tuple for axis-specific
                ``(D, H, W)`` stride.
            downsample: Optional residual projection applied to the identity path
                before residual addition. This is typically used when the spatial
                stride changes or when ``in_channels`` differs from ``out_channels``.

        Returns:
            None.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        """Run the residual block on a 3D feature tensor.

        Args:
            x: Input tensor with shape ``[N, C, D, H, W]``. ``C`` must match
                ``in_channels`` passed to the constructor.

        Returns:
            Output tensor with shape ``[N, out_channels, D_out, H_out, W_out]``.
            The spatial dimensions depend on the first-convolution stride and any
            residual projection supplied through ``downsample``.
        """
        identity: Tensor = x

        out: Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

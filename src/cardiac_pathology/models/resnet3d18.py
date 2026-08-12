"""Custom anisotropic 3D ResNet-18 classifier for cardiac MRI tensors."""

from torch import Tensor, nn

from cardiac_pathology.models.basic_block_3d import BasicBlock3D

Stride3D = tuple[int, int, int]


class ResNet3D18(nn.Module):
    """Anisotropic 3D ResNet-18 for patient-level cardiac MRI classification.

    The network expects 5D PyTorch tensors with shape ``[N, C, D, H, W]``. For
    the frozen preprocessing contract, inputs have shape ``[N, 2, 14, 144, 144]``,
    where ``D``, ``H``, and ``W`` represent ``Z``, ``Y``, and ``X`` respectively.
    The architecture preserves depth through the stem, max-pooling, and first
    three residual stages, then performs the only depth downsampling in ``layer4``.
    """

    def __init__(self, input_channels: int = 2, num_classes: int = 5) -> None:
        """Initialize the custom anisotropic 3D ResNet-18 classifier.

        Args:
            input_channels: Number of input channels in tensors with shape
                ``[N, C, D, H, W]``. The frozen preprocessing contract uses two
                channels: ED and ES.
            num_classes: Number of output class logits. The ACDC classification
                task uses five pathology classes.

        Returns:
            None.
        """
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv3d(
            input_channels,
            self.in_channels,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

        self.layer1 = self._make_layer(out_channels=64, block_count=2, stride=(1, 1, 1))
        self.layer2 = self._make_layer(out_channels=128, block_count=2, stride=(1, 2, 2))
        self.layer3 = self._make_layer(out_channels=256, block_count=2, stride=(1, 2, 2))
        self.layer4 = self._make_layer(out_channels=512, block_count=2, stride=(2, 2, 2))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * BasicBlock3D.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        block_count: int,
        stride: Stride3D,
    ) -> nn.Sequential:
        """Build one residual stage from ``BasicBlock3D`` blocks.

        Args:
            out_channels: Number of channels produced by the residual stage.
            block_count: Number of basic residual blocks in the stage.
            stride: Stride for the first block in ``[D, H, W]`` order. Use
                ``(1, 2, 2)`` to preserve depth while downsampling in-plane axes,
                and ``(2, 2, 2)`` for the final stage depth downsampling.

        Returns:
            Sequential residual stage that maps input tensors from
            ``[N, in_channels, D, H, W]`` to
            ``[N, out_channels, D_out, H_out, W_out]``.
        """
        downsample: nn.Module | None = None
        expanded_channels = out_channels * BasicBlock3D.expansion
        if stride != (1, 1, 1) or self.in_channels != expanded_channels:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    expanded_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(expanded_channels),
            )

        blocks: list[nn.Module] = [
            BasicBlock3D(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.in_channels = expanded_channels
        for _index in range(1, block_count):
            blocks.append(
                BasicBlock3D(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                )
            )

        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        """Initialize convolution, normalization, and classifier parameters.

        Conv3d weights use Kaiming normal initialization for ReLU activations.
        BatchNorm3d affine parameters are initialized to weight ``1`` and bias
        ``0``. Linear classifier weights use a small zero-mean normal
        distribution and classifier bias is initialized to ``0``.

        Returns:
            None.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Compute raw class logits from a preprocessed cardiac MRI batch.

        Args:
            x: Input tensor with shape ``[N, input_channels, D, H, W]``. Under the
                frozen preprocessing contract this is ``[N, 2, 14, 144, 144]``,
                with spatial axes ``[D, H, W] = [Z, Y, X]``.

        Returns:
            Raw, unnormalized class logits with shape ``[N, num_classes]``. No
            softmax or other probability normalization is applied.
        """
        out: Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.flatten(start_dim=1)
        out = self.fc(out)

        return out

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ConditionalDeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ wrapper that conditions on a vector via FiLM at the encoder bottleneck.

    Usage:
        model = ConditionalDeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            vec_dim=10,
        )
        y = model(x, vec)   # x: (B,C,H,W), vec: (B,vec_dim)
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        vec_dim: int = 10,
        **dlab_kwargs,
    ):
        super().__init__()

        # Core SMP model
        self.net = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **dlab_kwargs,
        )

        # Channels of the deepest encoder feature (bottleneck)
        bottleneck_ch: int = self.net.encoder.out_channels[-1]

        # Small FiLM MLP: vec -> (gamma, beta) for bottleneck channels
        hidden = max(128, bottleneck_ch // 2)
        self.film = nn.Sequential(
            nn.Linear(vec_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * bottleneck_ch),
        )
        # Identity init so initial conditioning is a no-op (gamma≈0, beta≈0)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, x: torch.Tensor, vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:   (B, C, H, W)
        vec: (B, vec_dim) or None (no conditioning)
        """
        # Encode
        feats = self.net.encoder(x)  # list of feature maps at multiple strides

        if vec is not None:
            bneck = feats[-1]  # deepest feature
            B, Cb, H, W = bneck.shape

            if vec.dim() == 1:
                vec = vec.unsqueeze(0)
            assert vec.shape[0] == B, "Batch size mismatch between x and vec."

            gamma_beta = self.film(vec)  # (B, 2*Cb)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            gamma = gamma.view(B, Cb, 1, 1).to(dtype=bneck.dtype)
            beta = beta.view(B, Cb, 1, 1).to(dtype=bneck.dtype)

            # FiLM: scale-and-shift with residual-friendly scaling (1 + gamma)
            feats[-1] = bneck * (1 + gamma) + beta

        # Decode & segment
        dec = self.net.decoder(feats)
        out = self.net.segmentation_head(dec)
        return out

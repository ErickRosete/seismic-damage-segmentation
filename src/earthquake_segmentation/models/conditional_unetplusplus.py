from __future__ import annotations
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ConditionalUnetPlusPlus(nn.Module):
    """
    Unet++ wrapper that conditions on a vector via FiLM at the encoder bottleneck.

    Example:
        model = ConditionalUnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            vec_dim=10,
        )
        y = model(x, vec)  # x: (B,C,H,W), vec: (B,vec_dim)
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        vec_dim: int = 10,
        **unetpp_kwargs,
    ):
        super().__init__()

        # Core SMP model
        self.net = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **unetpp_kwargs,
        )

        # Bottleneck channels (deepest encoder feature)
        bottleneck_ch = self.net.encoder.out_channels[-1]

        # FiLM MLP: vec -> (gamma, beta)
        hidden = max(128, bottleneck_ch // 2)
        self.film = nn.Sequential(
            nn.Linear(vec_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * bottleneck_ch),
        )
        # Identity-ish init: start as near no-op conditioning
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        feats = self.net.encoder(x)  # list of multi-scale features
        bneck = feats[-1]
        B, Cb, H, W = bneck.shape

        if vec.dim() == 1:
            vec = vec.unsqueeze(0)
        assert vec.shape[0] == B, "Batch size mismatch between image and condition vector."

        gamma_beta = self.film(vec)  # (B, 2*Cb)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(B, Cb, 1, 1).to(dtype=bneck.dtype, device=bneck.device)
        beta = beta.view(B, Cb, 1, 1).to(dtype=bneck.dtype, device=bneck.device)

        feats[-1] = bneck * (1 + gamma) + beta

        dec = self.net.decoder(feats)              # Unpack features
        out = self.net.segmentation_head(dec)
        return out

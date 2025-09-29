import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ConditionalUNetModel(nn.Module):
    """UNet wrapper with FiLM at the bottleneck."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        vec_dim: int = 10,
        **unet_kwargs,
    ):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **unet_kwargs,
        )

        bottleneck_ch = self.unet.encoder.out_channels[-1]
        hidden = max(128, bottleneck_ch // 2)
        self.film_mlp = nn.Sequential(
            nn.Linear(vec_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * bottleneck_ch),
        )
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        feats = self.unet.encoder(x)
        bneck = feats[-1]
        B, Cb, H, W = bneck.shape

        gamma_beta = self.film_mlp(vec)  # (B, 2*Cb)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(B, Cb, 1, 1).to(dtype=bneck.dtype)
        beta = beta.view(B, Cb, 1, 1).to(dtype=bneck.dtype)

        feats[-1] = bneck * (1 + gamma) + beta
        dec_out = self.unet.decoder(feats)
        masks = self.unet.segmentation_head(dec_out)
        return masks

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ConditionalUNetModel(nn.Module):
    """
    A wrapper around segmentation_models_pytorch.Unet that
    conditions on an extra vector injected at the bottleneck.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        vec_dim: int = 10,
        **unet_kwargs,
    ):
        super().__init__()
        # -- core Unet --
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **unet_kwargs,
        )

        # channels at the deepest encoder feature
        bottleneck_ch = self.unet.encoder.out_channels[-1]

        self.film_mlp = nn.Sequential(
            nn.Linear(vec_dim, bottleneck_ch * 2),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_ch * 2, bottleneck_ch * 2),
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        # Encode image into feature maps
        feats = self.unet.encoder(x)  # list of 5 feature maps
        bneck = feats[-1]  # the deepest one
        B, Cb, H, W = bneck.shape

        # FiLM parameters
        gamma_beta = self.film_mlp(vec)  # (B, 2*Cb)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        # apply FiLM
        gamma = gamma.view(B, Cb, 1, 1)
        beta = beta.view(B, Cb, 1, 1)
        feats[-1] = bneck * (1 + gamma) + beta

        # Decode + head as usual
        dec_out = self.unet.decoder(feats)
        masks = self.unet.segmentation_head(dec_out)
        return masks

import segmentation_models_pytorch as smp
from earthquake_segmentation.models.conditional_unet import ConditionalUNetModel


def build_model(cfg):
    """
    Build segmentation model based on config.
    """
    m = cfg.model

    # Classic segmentation models
    if m.name == "unet":
        return smp.Unet(
            encoder_name=m.encoder,
            encoder_weights=m.encoder_weights,
            in_channels=3,
            classes=m.num_classes,
        )

    if m.name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=m.encoder,
            encoder_weights=m.encoder_weights,
            in_channels=3,
            classes=m.num_classes,
        )
    if m.name == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name=m.encoder,
            encoder_weights=m.encoder_weights,
            in_channels=3,
            classes=m.num_classes,
        )

    # Conditional models
    if m.name == "conditional_unet":
        return ConditionalUNetModel(
            encoder_name=m.encoder,
            encoder_weights=m.encoder_weights,
            in_channels=3,
            classes=m.num_classes,
            vec_dim=m.vec_dim,
        )

    # Add additional architectures here
    raise ValueError(f"Unsupported model: {m.name}")

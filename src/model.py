import torch.nn as nn

from torchvision.models.video import (
    r3d_18,
    R3D_18_Weights,
    r2plus1d_18,
    R2Plus1D_18_Weights,
    mc3_18,
    MC3_18_Weights,
    S3D,
    S3D_Weights
)


def get_model(model_name="r3d18"):

    model_name = model_name.lower()

    if model_name == "r3d18":

        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)

    elif model_name == "r2plus1d18":

        weights = R2Plus1D_18_Weights.DEFAULT
        model = r2plus1d_18(weights=weights)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)

    elif model_name == "mc3_18":

        weights = MC3_18_Weights.DEFAULT
        model = mc3_18(weights=weights)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)

    elif model_name == "s3d":

        model = S3D(num_classes=400)

        weights = S3D_Weights.DEFAULT
        model.load_state_dict(weights.get_state_dict())

        # Make S3D compatible with our 16x112x112 input
        model.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        in_channels = model.classifier[1].in_channels

        model.classifier[1] = nn.Conv3d(
            in_channels,
            2,
            kernel_size=1
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from: r3d18, r2plus1d18, mc3_18, s3d"
        )

    return model
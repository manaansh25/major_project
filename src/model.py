import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

def get_model():
    weights = R3D_18_Weights.DEFAULT

    model = r3d_18(weights=weights)

    in_features = model.fc.in_features

    model.fc = nn.Linear(in_features, 2)

    return model
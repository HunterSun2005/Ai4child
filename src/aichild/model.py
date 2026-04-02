from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import model as base_model


class MultiTaskEfficientGCN(nn.Module):
    def __init__(self, config: dict, graph):
        super().__init__()
        data_cfg = config["data"]
        model_cfg = config["model"]

        inputs = data_cfg["inputs"]
        num_input = len(inputs)
        num_frame = int(data_cfg["num_frame"])
        num_joint = len(data_cfg["keypoint_indices"])

        # C=2 (x,y) -> C*2 after J/V/B feature construction.
        num_channel = 4

        kwargs = {
            "data_shape": [num_input, num_channel, num_frame, num_joint, 1],
            "num_class": 1,
            "A": torch.tensor(graph.A, dtype=torch.float32),
            "parts": graph.parts,
        }
        kwargs.update(model_cfg["model_args"])

        self.backbone = base_model.create(model_cfg["model_type"], **kwargs)

        feature_dim = int(self.backbone.classifier.fc.in_channels)
        self.feature_dim = feature_dim
        self.direction_dim = 4
        fused_dim = feature_dim + self.direction_dim

        self.track1_left_head = nn.Linear(fused_dim, 17)
        self.track1_right_head = nn.Linear(fused_dim, 17)

        self.track2_left_head = nn.Linear(fused_dim, 5)
        self.track2_right_head = nn.Linear(fused_dim, 5)

        ssl_dim = int(config["train"]["ssl_proj_dim"])
        self.ssl_projector = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, ssl_dim),
        )

    def _fused_feature(self, x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        _, feature = self.backbone(x)
        pooled = feature.mean(dim=(2, 3, 4))
        return torch.cat([pooled, direction], dim=1)

    def forward(self, x: torch.Tensor, direction: torch.Tensor) -> Dict[str, torch.Tensor]:
        fused = self._fused_feature(x, direction)
        outputs = {
            "track1_left": self.track1_left_head(fused),
            "track1_right": self.track1_right_head(fused),
            "track2_left": self.track2_left_head(fused),
            "track2_right": self.track2_right_head(fused),
        }
        return outputs

    def ssl_embedding(self, x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        fused = self._fused_feature(x, direction)
        z = self.ssl_projector(fused)
        return F.normalize(z, dim=1)

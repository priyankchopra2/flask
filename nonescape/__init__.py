# Copyright 2025 Aedilic Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch
from torch import Tensor, nn
import torchvision.models as models
import torchvision.transforms as transforms
from safetensors.torch import load_file
from PIL import Image


def preprocess_image(image: Image.Image) -> Tensor:
    """Preprocess image for Nonescape models.

    Args:
        image: PIL Image

    Returns:
        Preprocessed tensor ready for model input
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)



class NonescapeClassifierMini(nn.Module):
    """EfficientNet-based fake image detector"""

    in_features = None  # Override

    def __init__(self, num_classes: int = 2, embedding_size: int = 1024, dropout: float = 0.2):
        super().__init__()

        self.backbone = models.efficientnet_v2_s(weights=None, num_classes=embedding_size, dropout=dropout)
        self.head = nn.Linear(embedding_size, num_classes)

    @classmethod
    def from_pretrained(cls, path: str) -> NonescapeClassifierMini:
        state_dict = load_file(path)

        model = cls()
        model.load_state_dict(state_dict)

        return model

    def forward(self, x: Tensor) -> Tensor:
        emb = self.backbone(x)
        logits = self.head(emb)
        probs = nn.functional.softmax(logits, dim=-1)

        return probs


__all__ = [ "NonescapeClassifierMini", "preprocess_image"]

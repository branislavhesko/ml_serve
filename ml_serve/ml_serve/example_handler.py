import base64

import cv2
import numpy as np
import torch
from torchvision.models.resnet import resnet18
from torchvision.models import ResNet18_Weights

from ml_housekeeper.ml_housekeeper.base_handler import BaseHandler


class ExampleHandler(BaseHandler):
    def __init__(self):
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model = self.model.to(self.device)

    def predict(self, image_data: bytes) -> dict[str, bytes]:
        image = self._decode_image(image_data["image"])
        image = self.transform(torch.from_numpy(image).permute(2, 0, 1))
        image = image.unsqueeze(0)
        image = image.to(self.device)
        output = self.model(image).squeeze(0)
        return {"prediction": output.detach().cpu().numpy().tolist()}

    def _decode_image(self, image_data: bytes) -> np.ndarray:
        return cv2.imdecode(np.array(image_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    def get_model_name(self) -> str:
        return "example_model"

    def get_model_version(self) -> str:
        return "1.0"

    def get_model_path(self) -> str:
        return "example_model"

    def get_num_workers(self) -> int:
        return 1

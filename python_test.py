import os

import grpc
from predict_pb2_grpc import InferenceStub
from predict_pb2 import PredictionsRequest, CreateModelWorkerRequest
from google.protobuf.empty_pb2 import Empty
import json
import numpy as np
from torchvision.models import ResNet18_Weights
classes = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]


def _encode_image(image_path: str) -> bytes:
    with open(image_path, "rb") as image_file:
        return image_file.read()


class InferenceClient:
    def __init__(self, host: str, port: int):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = InferenceStub(self.channel)

    def ping(self):
        response = self.stub.Ping(Empty())
        return response

    def predictions(self, model_name: str, model_version: str, input: dict[str, bytes]):
        response = self.stub.Predictions(PredictionsRequest(model_name=model_name, model_version=model_version, input=input))
        return response
    
    def create_model_worker(self, model_name: str, model_path: str, num_workers: int):
        response = self.stub.CreateModelWorker(CreateModelWorkerRequest(model_name=model_name, model_path=model_path, num_workers=num_workers))
        return response

try:
    client = InferenceClient("localhost", 50051)
    print(client.ping())
    print(client.create_model_worker("resnet18", os.path.join(os.path.dirname(__file__), "assets/example_model.mlkeep"), 1))
except Exception as e:
    print(e)

for i in range(10):
    output = client.predictions("resnet18", "1", {"image": _encode_image(os.path.join(os.path.dirname(__file__), "assets/snake.jpg"))}).prediction
output = json.loads(output["prediction"])
print(output)
print(np.array(output).shape)
top_k = 50
top_k_indices = np.argsort(output)[-top_k:][::-1]

for i, index in enumerate(top_k_indices):
    print(f"{i+1}. {classes[index]} ({output[index]:.2f})")

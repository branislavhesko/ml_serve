import argparse
import base64
from datetime import datetime
import importlib
import json
import os
import tempfile
import zipfile
import sys
sys.path.append("/home/brani/code/mlhousekeeper")

import cv2
import numpy as np
import zmq

from ml_housekeeper.ml_housekeeper.base_handler import BaseHandler
from ml_housekeeper.ml_housekeeper.example_handler import ExampleHandler


class MLHousekeeperServer:

    def __init__(self, model_file: str, host: str, port: int):
        self.model_file = model_file
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{self.host}:{self.port}")
        self.handler = None
        self._load_model()

    def _load_model(self):
        print(f"Loading model from {self.model_file}")
        assert os.path.exists(self.model_file), f"Model file {self.model_file} does not exist"
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(self.model_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            os.chdir(temp_dir)
            with open("manifest.json", "r") as f:
                manifest = json.load(f)
                print(manifest)
            handler_module = importlib.import_module(manifest["handler"])
            self.handler = getattr(handler_module, manifest["handler_class_name"])()

    def start(self):
        while True:
            try:
                self.handle_multipart_message(self.socket)
            except Exception as e:
                print(f"Error: {e}")
                self.socket.send(json.dumps({"error": str(e)}).encode('utf-8'))

    def handle_multipart_message(self, socket):
        """
        Handle multipart messages with separate parts for each field.
        """
        # Receive all parts
        parts = []
        while True:
            part = socket.recv()
            parts.append(part)
            if not socket.getsockopt(zmq.RCVMORE):
                break
        
        # Process parts
        response_text = parts[0].decode('utf-8')
        metadata_json = json.loads(parts[1].decode('utf-8'))
        image_data = json.loads(parts[2].decode('utf-8'))
        response = self.handler.predict(image_data)
        socket.send(json.dumps(response).encode('utf-8'))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    server = MLHousekeeperServer(args.model_file, args.host, args.port)
    server.start()
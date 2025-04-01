import dataclasses
import json
import os
import zipfile


@dataclasses.dataclass
class PackModelArgs:
    handler_path: str
    model_path: str
    config_path: str
    model_name: str
    handler_class_name: str
    files: list[str] = dataclasses.field(default_factory=list)


def pack_model(args: PackModelArgs, output_path: str):
    assert os.path.exists(args.handler_path), f"Handler path {args.handler_path} does not exist"
    assert os.path.exists(args.model_path), f"Model path {args.model_path} does not exist"
    assert os.path.exists(args.config_path), f"Config path {args.config_path} does not exist"
    assert os.path.exists(output_path), f"Output path {output_path} does not exist"
    
    with zipfile.ZipFile(os.path.join(output_path, f"{args.model_name}.mlkeep"), 'w') as zipf:
        zipf.write(args.handler_path, os.path.basename(args.handler_path))
        zipf.write(args.model_path, os.path.basename(args.model_path))
        zipf.write(args.config_path, os.path.basename(args.config_path))
        zipf.writestr("manifest.json", json.dumps(_make_manifest(args), indent=4))
        for file in args.files:
            zipf.write(file, os.path.basename(file))


def _make_manifest(args: PackModelArgs):
    manifest = {
        "model_name": args.model_name,
        "model_version": 1,
        "handler": os.path.basename(args.handler_path).replace(".py", ""),
        "handler_class_name": args.handler_class_name,
        "model": os.path.basename(args.model_path),
        "config": os.path.basename(args.config_path),
        "files": [os.path.basename(file) for file in args.files],
    }
    return manifest

if __name__ == "__main__":
    args = PackModelArgs(
        handler_path=os.path.join(os.path.dirname(__file__), "example_handler.py"),
        model_path=os.path.join(os.path.dirname(__file__), "resnet18-f37072fd.pth"),
        config_path=os.path.join(os.path.dirname(__file__), "example_config.yaml"),
        model_name="example_model",
        handler_class_name="ExampleHandler",
        files=[os.path.join(os.path.dirname(__file__), "base_handler.py")]
    )
    pack_model(args, "./")
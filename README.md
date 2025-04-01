> ⚠️ **Disclaimer**: This project is in early stages of development and is not production-ready. It is currently being used for experimental purposes and may contain bugs or undergo significant changes. Use at your own risk.


# Machine model serving in Rust


<img src="assets/logo.png" width="300px" style="display: block; margin: 0 auto;">


---


```
Serve your python models using Rust + GRPC service. Made for simplicity as a replacement of torchserve.
```


## Demo
Currently, only a single model is provided as a runnable demo. It can be modified to work with other Python code.

```bash
python -m grpc_tools.protoc -I=./proto --python_out=. --grpc_python_out=. ./proto/predict.proto
cargo run --release
```

And in separate terminal

```bash
python python_test.py
```


## Feedback
If you want any feature, please write me an email: `branislav@hesko.space`.
use tonic::{transport::Server, Request, Response, Status};
use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};
mod worker;
mod utils;

use worker::Worker;
use worker::Message;
use worker::start;
use predict::inference_server::{Inference, InferenceServer};
use predict::{
    PredictionsRequest, 
    PredictionResponse, 
    MlServantHealthResponse,
    CreateModelWorkerRequest, 
    CreateModelWorkerResponse
};


pub mod predict {
    tonic::include_proto!("predict");
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "0.0.0.0:50051".parse().unwrap();
    let inference_server =  InferenceModel::default();

    println!("MLServant is running on {}", addr);

    Server::builder()
        .add_service(InferenceServer::new(inference_server))
        .serve(addr)
        .await?;

    Ok(())
}


#[derive(Debug)]
pub struct InferenceWorker {
    pub worker: Arc<Mutex<Worker>>,
    pub input_pipe: mpsc::Sender<Message>,
    pub output_pipe: mpsc::Receiver<Message>,
}

impl InferenceWorker {
    pub fn new(model_name: String, model_path: String, host: String, zmq_port: u16) -> Self {
        let (input_pipe_sender, input_pipe_receiver) = mpsc::channel();
        let (output_pipe_sender, output_pipe_receiver) = mpsc::channel();
        let worker = Arc::new(Mutex::new(Worker::new(
            model_name, 
            host, 
            zmq_port, 
            model_path,
            input_pipe_receiver, 
            output_pipe_sender
        )));
        Self { worker, input_pipe: input_pipe_sender, output_pipe: output_pipe_receiver }
    }
}

#[derive(Debug, Default)]
pub struct InferenceModel {
    workers: Mutex<HashMap<String, InferenceWorker>>,
}


impl InferenceModel {
    pub fn new() -> Self {
        Self { workers: Mutex::new(HashMap::new()) }
    }
}


#[tonic::async_trait]
impl Inference for InferenceModel {
    async fn ping(&self, request: Request<()>) -> Result<Response<MlServantHealthResponse>, Status> {
        let _ = request;
        Ok(Response::new(MlServantHealthResponse { health: "OK".to_string() }))
    }

    async fn predictions(&self, request: Request<PredictionsRequest>) -> Result<Response<PredictionResponse>, Status> {
        // Lock the workers hashmap
        let workers = self.workers.lock().unwrap();
        let request = request.into_inner();
        println!("Predictions request for model: {:?}", request.model_name.clone());
        let model_name = request.model_name.clone();
        let worker = match workers.get(&model_name) {
            Some(w) => w,
            None => return Err(Status::not_found("Worker not found")),
        };
        println!("Sending message to worker");
        let _ = worker.input_pipe.send(Message::from(request.input));
        println!("Message sent to worker");
        let response = worker.output_pipe.recv_timeout(std::time::Duration::from_secs(5));   
        let response = match response {
            Ok(response) => response,
            Err(e) => return Err(Status::not_found("No response found after waiting")),
        };

        return Ok(Response::new(PredictionResponse { 
            prediction: response.data, 
            model_name: model_name,
            sequence_id: Some("123".to_string()),
        }));

    }

    async fn create_model_worker(&self, request: Request<CreateModelWorkerRequest>) -> Result<Response<CreateModelWorkerResponse>, Status> {
        let message = request.into_inner();
        let host: String = "localhost".to_string();
        let zmq_port: u16 = 5555;
        let model_name = message.model_name.clone();
        let model_path = message.model_path;
        if self.workers.lock().unwrap().contains_key(&model_name) {
            return Err(Status::already_exists("Worker already exists"));
        }
        let worker = InferenceWorker::new(model_name, model_path, host, zmq_port);
        let worker_clone = worker.worker.clone();
        std::thread::spawn(move || {
            let _ = start(worker_clone);
        });

        self.workers.lock().unwrap().insert(message.model_name.clone(), worker);

        Ok(Response::new(CreateModelWorkerResponse { status: "OK".to_string(), model_name: message.model_name.clone() }))
    }

}

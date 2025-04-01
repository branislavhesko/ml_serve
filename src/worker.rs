use std::os::unix::process::CommandExt;
use std::process::{Command, Child};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, atomic::AtomicBool, mpsc};
use std::fmt;
use std::time::Instant;
use rand::Rng;
use bincode;
use serde_json;
use zmq;
use zmq::{Context, Socket};

use crate::utils::{encode_image_base64, decode_image_base64};


#[derive(Debug)]
pub struct Message {
    pub data: HashMap<String, Vec<u8>>,
}

impl Message {
    pub fn new(data: HashMap<String, Vec<u8>>) -> Self {
        Self { data }
    }

    pub fn from_hashmap(data: HashMap<String, Vec<u8>>) -> Self {
        Self { data }
    }
}

impl From<HashMap<String, Vec<u8>>> for Message {
    fn from(data: HashMap<String, Vec<u8>>) -> Self {
        Self::from_hashmap(data)
    }
}


impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Message {{ data: {:?} }}", self.data)
    }
}


pub struct Worker {
    model_name: String,
    host: String,
    zmq_port: u16,
    model_path: String,
    python_process_handler: Option<Child>,
    pub input_pipe: mpsc::Receiver<Message>,
    pub output_pipe: mpsc::Sender<Message>,
    pub stop_event: Arc<AtomicBool>,
    pub context: Context,
    pub socket: Socket,
}

impl fmt::Debug for Worker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Worker")
            .field("model_name", &self.model_name)
            .field("host", &self.host)
            .field("zmq_port", &self.zmq_port)
            .field("model_path", &self.model_path)
            .field("python_process_handler", &self.python_process_handler)
            .field("input_pipe", &self.input_pipe)
            .field("output_pipe", &self.output_pipe)
            .field("stop_event", &self.stop_event)
            .finish_non_exhaustive()
    }
}


impl Worker {
    pub fn new(model_name: String, host: String, zmq_port: u16, model_path: String, input_pipe: mpsc::Receiver<Message>, output_pipe: mpsc::Sender<Message>) -> Self {
        let stop_event = Arc::new(AtomicBool::new(false));
        let context = Context::new();
        let socket = context.socket(zmq::REQ).expect("Failed to create socket");
        socket.connect(&format!("tcp://{}:{}", host, zmq_port)).expect("Failed to connect to socket");
        println!("Connected to socket");
        Self { 
            model_name, 
            host, 
            zmq_port, 
            model_path,
            python_process_handler: None,
            input_pipe,
            output_pipe,
            stop_event,
            context,
            socket
        }
    }

    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Running worker");
        let command = Command::new("python")
            .arg("ml_serve/ml_serve/server.py")
            .arg("--model_file")
            .arg(&self.model_path)
            .spawn()?;
        println!("Python process started");
        self.python_process_handler = Some(command);
        Ok(())
    }


    pub fn send_message(&self, message: Message) -> Result<Message, Box<dyn std::error::Error>> {
        let metadata = HashMap::from([("response".to_string(), "Hello".to_string()), ("metadata".to_string(), "Hello".to_string())]);
        let response_text = "Hello";
        let response = send_using_multipart(&self.context, &message.data, &response_text, &metadata)?;
        Ok(response)
    }

    pub fn receive_message(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}


pub fn send_using_multipart(
    context: &zmq::Context, 
    image_data: &HashMap<String, Vec<u8>>, 
    response_text: &str, 
    metadata: &HashMap<String, String>
) -> Result<Message, Box<dyn std::error::Error>> {
    // Create a REQ (request) socket
    let socket = context.socket(zmq::REQ).expect("Failed to create socket");
    
    socket.connect("tcp://127.0.0.1:5555").expect("Failed to connect");
    println!("Connected to server (multipart method)");
    
    let start = Instant::now();
    
    let metadata_json = serde_json::to_string(metadata).expect("Failed to serialize metadata");
    
    let msg = zmq::Message::from(response_text.as_bytes());
    socket.send(msg, zmq::SNDMORE).expect("Failed to send response part");
    
    let msg = zmq::Message::from(metadata_json.as_bytes());
    socket.send(msg, zmq::SNDMORE).expect("Failed to send metadata part");
    let json_data = serde_json::to_string(image_data).expect("Failed to serialize image data");
    let msg = zmq::Message::from(json_data.as_bytes());
    socket.send(msg, 0).expect("Failed to send image part");
    
    // Receive the response
    let mut buffer = zmq::Message::new();
    socket.recv(&mut buffer, 0).expect("Failed to receive response");
    println!("Sent message using multipart, duration: {:?}", start.elapsed());    

    let mut data: HashMap<String, Vec<u8>> = HashMap::new();
    let response: HashMap<String, serde_json::Value> = serde_json::from_slice(&buffer.as_ref())?;
    for (key, value) in response.iter() {
        let bytes = match value {
            serde_json::Value::String(s) => s.as_bytes().to_vec(),
            serde_json::Value::Number(n) => n.to_string().into_bytes(),
            serde_json::Value::Array(arr) => serde_json::to_string(arr)?.into_bytes(),
            serde_json::Value::Object(obj) => serde_json::to_string(obj)?.into_bytes(),
            serde_json::Value::Bool(b) => b.to_string().into_bytes(),
            serde_json::Value::Null => Vec::new(),
        };
        data.insert(key.to_string(), bytes);
    }
    Ok(Message::new(data))
}


pub fn stop(worker_guard: Arc<Mutex<Worker>>) -> Result<(), Box<dyn std::error::Error>> {
    let mut worker = worker_guard.lock().unwrap();
    if let Some(mut child) = worker.python_process_handler.take() {
        child.kill()?;
        return Ok(());
    }
    Err("Python process not found".into())
}


pub fn start(worker_guard: Arc<Mutex<Worker>>) -> Result<(), Box<dyn std::error::Error>> {
    start_new_python_server(&worker_guard)?;

    loop {
        std::thread::sleep(std::time::Duration::from_millis(10));
        {
            let worker = worker_guard.lock().unwrap();            
            if let Ok(message) = worker.input_pipe.recv_timeout(std::time::Duration::from_millis(100)) {
                let response = worker.send_message(message);
                if let Ok(response) = response {
                    let _ = worker.output_pipe.send(response);
                } else {
                    let _ = worker.output_pipe.send(Message::new(HashMap::from([("error".to_string(), "Error sending message".to_string().as_bytes().to_vec())])));
                }
            }
    
            if worker.stop_event.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
        }
    }
    Ok(())
}

fn start_new_python_server(worker_guard: &Arc<Mutex<Worker>>) -> Result<(), Box<dyn std::error::Error>> {
    let mut worker = worker_guard.lock().unwrap();
    let _ = worker.run()?;
    println!("Worker started");
    Ok(())
}

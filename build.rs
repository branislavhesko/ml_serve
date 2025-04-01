use std::env;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("proto/predict.proto")?;
    Ok(())
}

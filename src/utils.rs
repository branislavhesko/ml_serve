use base64;


pub fn encode_image_base64(image: &[u8]) -> String {
    let base64_image = base64::encode(image);
    base64_image
}

pub fn decode_image_base64(base64_image: &str) -> Vec<u8> {
    let image = base64::decode(base64_image).unwrap();
    image
}

import os, io, json, base64
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from utility.download_sam2 import ensure_sam2_weights

CONFIG_FILE = "config/sam2_config.yaml"

def model_fn(model_dir=None):
    ckpt_path, config_path, device = ensure_sam2_weights(CONFIG_FILE)
    print("Building SAM2 model...")
    model = build_sam2(config_path, ckpt_path, device=device)
    model.eval()
    return model

def input_fn(request_body, content_type='application/json'):
    payload = json.loads(request_body)
    img_bytes = base64.b64decode(payload['image'])
    image = Image.open(io.BytesIO(img_bytes))
    return image

def predict_fn(input_data, model):
    # Convert PIL -> tensor and run inference
    # Ensure float32, 0-1 range, 3 channels, channel-first, batch dim, and device match model
    img = np.array(input_data).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # NCHW
    device = next(model.parameters()).device if any(p.numel() for p in model.parameters()) else torch.device("cpu")
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        result = model(img_tensor)
    return result

def output_fn(prediction, accept='application/json'):
    # Convert tensor to JSON-safe output
    output = prediction[0].detach().cpu().numpy().tolist()
    return json.dumps({"mask": output})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SAM2 inference locally")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="mask.npy", help="Output path (.npy or .png/.jpg)")
    args = parser.parse_args()

    # Build model
    model = model_fn()

    # Load image
    image = Image.open(args.image).convert("RGB")

    # Predict
    prediction = predict_fn(image, model)

    # Extract mask tensor -> numpy
    mask = prediction[0].detach().cpu().numpy()

    # Save as .npy
    if args.output.lower().endswith((".npy", ".npz")):
        np.save(args.output, mask)
    else:
        # convert to 0-255 uint8 for image formats
        arr = mask
        # If mask has channel dimension, collapse or take first channel
        if arr.ndim == 3 and arr.shape[0] in (1,3):
            # if channel-first, move to HWC
            if arr.shape[0] in (1,3):
                arr = np.transpose(arr, (1,2,0))
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        img = Image.fromarray((arr * 255).astype(np.uint8))
        img.save(args.output)

    print("Saved mask to", args.output)
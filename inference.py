import os, io, json, base64
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from utils.download_sam2 import ensure_sam2_weights

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
    transform = torch.nn.Identity()  # placeholder; SAM2 may require preprocessing
    img_tensor = transform(torch.tensor(np.array(input_data))).permute(2, 0, 1).float().unsqueeze(0)
    
    with torch.no_grad():
        result = model(img_tensor)
    return result

def output_fn(prediction, accept='application/json'):
    # Convert tensor to JSON-safe output
    output = prediction[0].detach().cpu().numpy().tolist()
    return json.dumps({"mask": output})


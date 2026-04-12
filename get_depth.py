import torch
import cv2
import numpy as np
import os
from PIL import Image
import tqdm

# ======= 1. Load MiDaS model =======
print("Loading MiDaS model...")
midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')  # Best model for quality
midas.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
midas.to(device)

# ======= 2. Define custom transform =======
def custom_depth_transform(img):
    """
    Normalize image to [0,1] and prepare tensor.
    """
    img = img / 255.0  # Normalize to 0-1
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # From (H,W,C) to (C,H,W)
    return {"image": img}

# ======= 3. Setup input and output directories =======
input_dir = "train"      # Input folder
output_dir = "output_depths"    # Output folder

os.makedirs(output_dir, exist_ok=True)

# ======= 4. Get all image files =======
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_extensions]

# ======= 5. Loop through images =======
for filename in tqdm.tqdm(image_files, desc="Processing Images"):
    input_path = os.path.join(input_dir, filename)
    output_filename = os.path.splitext(filename)[0] + "_depth.png"
    output_path = os.path.join(output_dir, output_filename)

    # Load image
    img = np.array(Image.open(input_path).convert('RGB'))

    # Apply custom transform
    transformed = custom_depth_transform(img)
    img_input = transformed["image"].unsqueeze(0).to(device)  # Add batch dimension

    # Predict depth
    with torch.no_grad():
        prediction = midas(img_input)

    # Resize prediction to original size
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Clip and normalize depth
    depth_map_clipped = np.clip(depth_map, 0, 3000)  # optional range
    depth_map_16bit = (depth_map_clipped / 3000.0 * 65535.0).astype(np.uint16)

    # Save depth map
    cv2.imwrite(output_path, depth_map_16bit)

print("✅ All depth maps saved successfully!")

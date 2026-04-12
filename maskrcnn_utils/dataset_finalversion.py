import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


from torch.utils.data import Dataset, DataLoader
import json
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F




def convert_to_silver(img_path, intensity=0.8):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found by convert_to_silver: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Define color ranges for blue/green components
    blue_lower = np.array([90, 70, 50])
    blue_upper = np.array([130, 255, 255])
    green_lower = np.array([40, 70, 50])
    green_upper = np.array([80, 255, 255])

    # Create masks
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    combined_mask = cv2.bitwise_or(mask_blue, mask_green)

    # Create metallic silver texture
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    silver_base = np.full_like(img, 192) # Base silver color (RGB 192,192,192)

    # Preserve texture using luminosity blend
    silver_texture = cv2.addWeighted(
        silver_base, 0.7,
        np.dstack([gray]*3), 0.3,
        0
    )

    # Apply metallic effect only to masked areas
    result = img.copy()
    result[combined_mask > 0] = silver_texture[combined_mask > 0]

    # Enhance contrast
    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge((l_enhanced, a, b))
    final = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    return final


def convert_all_colored_to_silver(img_numpy, intensity=0.85): # Changed img_path to img_numpy
    # Assumes img_numpy is already loaded and in RGB format
    # If it's a path, you'd need to load it like: img = cv2.imread(img_path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV for better color isolation
    hsv = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2HSV)

    # Define broader color ranges (green, yellow, blue)
    color_ranges = [
        (np.array([20, 50, 50]), np.array([35, 255, 255])),  # Yellow
        (np.array([36, 50, 50]), np.array([85, 255, 255])),  # Green
        (np.array([86, 50, 50]), np.array([130, 255, 255]))]  # Blue

    # Create combined mask
    combined_mask = np.zeros(img_numpy.shape[:2], dtype=np.uint8)
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Enhance mask to catch adjacent colors
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # Create realistic metallic texture
    gray = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2GRAY)
    silver_base = np.full_like(img_numpy, (192, 192, 192))  # Base silver color

    # Add texture using luminosity blending
    silver_texture = cv2.addWeighted(
        silver_base, intensity,
        np.dstack([gray]*3), 1-intensity,
        0
    )

    # Apply metallic effect
    result = img_numpy.copy()
    result[combined_mask > 0] = silver_texture[combined_mask > 0]

    # Post-processing for enhanced contrast
    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l_channel)
    final = cv2.merge((l_enhanced, a, b))
    final = cv2.cvtColor(final, cv2.COLOR_LAB2RGB)

    return final

def add_random_stars(image_np, num_stars=100, star_brightness=255, star_size_range=(1, 3)):
    starred_image = image_np.copy()
    h, w, _ = starred_image.shape
    for _ in range(num_stars):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        radius = np.random.randint(star_size_range[0], star_size_range[1] + 1)
        cv2.circle(starred_image, (x, y), radius, (star_brightness, star_brightness, star_brightness), -1)
    return starred_image

class MaskDataset(Dataset):
   
    def __init__(self, coco_file, image_folder, mask_folder, transform=None,
                 add_stars=True, star_count_range=(20, 300), star_brightness=255, star_size_range=(1, 3)):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.add_stars=add_stars
        with open(coco_file, 'r') as file:
            coco_data = json.load(file)
        self.num_stars=9
        self.star_count_range = star_count_range # New: range for number of stars
        self.star_brightness = star_brightness
        self.star_size_range = star_size_range
        self.annotations = coco_data['annotations']
        self.images = {img['id']: img for img in coco_data['images']}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        image_info = self.images[image_id]

        # Load image using OpenCV
        image_path = os.path.join(self.image_folder, image_info['file_name'])
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Ensure RGB for consistency

        original_height, original_width = image.shape[:2] # OpenCV dimensions are H, W

        # Load mask
        mask_filename = f"{os.path.splitext(image_info['file_name'])[0]}.jpg"
        mask_path = os.path.join(self.mask_folder, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Get bounding box (COCO format: [x_min, y_min, width, height])
        bbox_coco = annotation['bbox']
        x1_orig, y1_orig, w_orig, h_orig = bbox_coco
        x2_orig, y2_orig = x1_orig + w_orig, y1_orig + h_orig

        # Convert mask and image to desired size (e.g., 640x640 for consistency)
        target_size = (640, 640) # W, H
        image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST) # Use nearest for masks
        # --- Add Stars to Background (BEFORE applying transforms if transforms normalize) ---
        image_resized=convert_all_colored_to_silver(image_resized)

        #print(mask_resized)
        if self.add_stars:

            #foreground_mask = mask_resized > 0

            # Create an inverted mask for the background
            #background_mask = ~foreground_mask

            num_stars_for_this_sample = np.random.randint(self.star_count_range[0], self.star_count_range[1] + 1)
            
            foreground_mask = mask_resized > 0
            background_mask = ~foreground_mask

            stars_canvas = np.zeros_like(image_resized)
            stars_canvas = add_random_stars(
                stars_canvas,
                num_stars=num_stars_for_this_sample, # Use the random count
                star_brightness=self.star_brightness,
                star_size_range=self.star_size_range
            )
            image_with_stars = image_resized.copy()
            
            # Apply stars_canvas pixels only where background_mask is True
            # This ensures stars don't appear on the object itself
            image_with_stars[background_mask] = stars_canvas[background_mask]
            
            image_resized = image_with_stars # Use this image for further processing
        # --- End Add Stars ---

        # Scale bounding box coordinates to the resized image dimensions
        xmin_scaled = x1_orig * (target_size[0] / original_width)
        ymin_scaled = y1_orig * (target_size[1] / original_height)
        xmax_scaled = x2_orig * (target_size[0] / original_width)
        ymax_scaled = y2_orig * (target_size[1] / original_height)
        
        boxes = torch.tensor([[xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled]], dtype=torch.float32)

        # Convert to tensors (using F.to_tensor expects PIL Image or NumPy array [H,W,C])
        # Image is a NumPy array [H,W,C] here, F.to_tensor converts it to [C,H,W] and normalizes to [0,1]
        image_tensor = F.to_tensor(image_resized)
        
        mask_tensor = torch.tensor(mask_resized, dtype=torch.uint8)
        mask_tensor = (mask_tensor > 0).unsqueeze(0) # Binary, add batch dimension

        target = {
            "boxes": boxes,
            "labels": torch.tensor([annotation['category_id']], dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "masks": mask_tensor,
            "area": torch.tensor([annotation['area']], dtype=torch.float32),
            "iscrowd": torch.tensor([annotation.get('iscrowd', 0)], dtype=torch.int64),
        }

        if self.transform:
            image_tensor = self.transform(image_tensor) # Apply transforms to the tensor

        return image_tensor, target




import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
class MaskRCNNDatasetFromCSV(Dataset):
    def __init__(self, csv_path, image_dir, mask_dir, transforms=None, mask_ext=".jpg"):
        self.data = pd.read_csv(csv_path, header=None)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.mask_ext = mask_ext

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image
        img_filename = row[0].strip()
        # Handle cases where CSV path might contain full paths like '/content/sunlamp_images/'
        # Ensure img_filename is just the basename if your image_dir is the base
        if "/content/sunlamp_images/" in img_filename:
            img_filename = img_filename.replace("/content/sunlamp_images/", "")
        
        image_path = os.path.join(self.image_dir, img_filename)
        
        # Check if the image path exists
        if not os.path.exists(image_path):
            # Fallback for common path issues if the image isn't found directly
            # This is a heuristic based on your commented-out lines
            if "/content/speedplusbaseline/" in image_path:
                image_path = image_path.replace("/content/speedplusbaseline/", "")
            
            # Try again with just the basename
            if not os.path.exists(image_path):
                 image_path = os.path.join(self.image_dir, os.path.basename(img_filename))
                 if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found at expected paths: {image_path} or original {os.path.join(self.image_dir, img_filename)}")

        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        # Desired output size for model input
        target_size = (640, 640)
        image_resized = image.resize(target_size)
        resized_width, resized_height = image_resized.size
        image_tensor = F.to_tensor(image_resized)

        # Bounding box (in original image size coordinates)
        xmin_orig, xmax_orig, ymin_orig, ymax_orig = float(row[1]), float(row[2]), float(row[3]), float(row[4])

        # Scale bounding box coordinates to the resized image dimensions
        # Note: Your original code normalized by width_n/height_n, but it should be
        # xmin / original_width * resized_width (which simplifies to xmin * (resized_width / original_width))
        xmin_scaled = xmin_orig * (resized_width / original_width)
        ymin_scaled = ymin_orig * (resized_height / original_height)
        xmax_scaled = xmax_orig * (resized_width / original_width)
        ymax_scaled = ymax_orig * (resized_height / original_height)
        
        boxes = torch.tensor([[xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled]], dtype=torch.float32)

        # Load mask
        base_name_without_ext = os.path.splitext(os.path.basename(img_filename))[0]
        mask_filename = base_name_without_ext + self.mask_ext
        mask_path = os.path.join(self.mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = Image.open(mask_path).convert("L") # grayscale
        mask = mask.resize(target_size)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)  # binary mask: 0 or 1
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        masks = mask.unsqueeze(0) # Add batch dimension [N, H, W]

        # Area and other target information
        area = torch.tensor([(xmax_orig - xmin_orig) * (ymax_orig - ymin_orig)], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64) # Assuming a single class '1'
        image_id = torch.tensor([idx])
        iscrowd = torch.tensor([0], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "masks": masks
        }

        # Apply transformations (like ToTensor) LAST for visualization
        if self.transforms:
            image_tensor = self.transforms(image_tensor)

        return image_tensor, target

# Define the basic transform for MaskRCNNDatasetFromCSV
# Use ToTensor for model input, but for visualization, we might convert back to numpy
transform_csv = transforms.Compose([
    transforms.ToTensor() # Converts PIL Image to FloatTensor in [0.0, 1.0]
])

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class MaskRCNNDatasetFromCSV(Dataset):
    def __init__(self, csv_path, image_dir, mask_dir,transforms=None, mask_ext=".png"):
        self.data = pd.read_csv(csv_path, header=None)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.mask_ext = mask_ext  # in case masks are .png while images are .jpg


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image
        img_filename = row[0].strip()
        img_filename=img_filename.replace("/content/sunlamp_images/", "")

        image_path = os.path.join(self.image_dir, img_filename)
        #image_path=image_path.replace("/content/sunlamp_images/", "")
                                          #image_path.replace("/content/speedplusbaseline/", "")
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        # Resize image to 500x500
        image = image.resize((640, 640))
        width_n, height_n = image.size

        # Bounding box (in original image size)
        xmin, xmax, ymin, ymax = float(row[1]), float(row[2]), float(row[3]), float(row[4])

        # Create binary mask from bounding box
        #mask = np.zeros((height_n, width_n), dtype=np.uint8)
        x1 = int(xmin / width * 640)
        x2 = int(xmax / width * 640)
        y1 = int(ymin / height * 640)
        y2 = int(ymax / height * 640)
        #mask[y1:y2, x1:x2] = 1
        #mask = torch.as_tensor(mask, dtype=torch.uint8)




        # Add batch dimension [N, H, W] (if 1 object per image)
        #masks = mask.unsqueeze(0)
        base_name = os.path.split(img_filename)[1]
        #print(base_name)
        mask_filename = base_name
        mask_path = os.path.join(self.mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        #print(mask_path)

        mask = Image.open(mask_path).convert("L")  # grayscale
        mask = mask.resize((640, 640))
        # Binarize mask
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)  # binary mask: 0 or 1
        mask = torch.as_tensor(mask, dtype=torch.uint8)

        # Add batch dimension [N, H, W] (if 1 object per image)
        masks = mask.unsqueeze(0)
        # Area before normalization

        # Add batch dimension [N, H, W] (if 1 object per image)
        #masks = mask.unsqueeze(0)

        # Area before normalization
        area = torch.tensor([(xmax - xmin) * (ymax - ymin)], dtype=torch.float32)

        # Normalize coordinates according to resized image dimensions
        xmin = xmin / width_n
        xmax = xmax / width_n
        ymin = ymin / height_n
        ymax = ymax / height_n
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

        # Labels, image_id, iscrowd
        labels = torch.tensor([1], dtype=torch.int64)
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

        if self.transforms:
            image = self.transforms(image)

        return image, target



from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

#csv_path = "sunlamp_test.csv"
#image_dir = "sun_lamp/images/"
#mask_folder = "sunlamp/masks/"

#dataset = MaskRCNNDatasetFromCSV(csv_path, image_dir, mask_dir=mask_folder,transforms=transform, mask_ext=".jpg")

#dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
#image=dataset[0][0],dataset[0][1]['masks']



import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class MaskDataset(Dataset):
    def __init__(self, coco_file, image_folder, mask_folder, transform=None):
        """
        Args:
            coco_file (str): Path to the COCO-style JSON file.
            image_folder (str): Folder containing the images.
            mask_folder (str): Folder containing pre-generated masks.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform

        # Load COCO annotations
        with open(coco_file, 'r') as file:
            coco_data = json.load(file)

        self.annotations = coco_data['annotations']
        self.images = {img['id']: img for img in coco_data['images']}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        image_info = self.images[image_id]

        # Load image
        image_path = os.path.join(self.image_folder, image_info['file_name'])
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_filename = f"{image_info['file_name'].split('.')[0]}_{annotation['id']}.png"
        mask_path = os.path.join(self.mask_folder, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Get bounding box
        bbox = annotation['bbox']  # COCO format: [x_min, y_min, width, height]
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h

        # Normalize bounding box based on original image size
        original_width, original_height = image.shape[1], image.shape[0]
        bbox = [
            x1 / original_width,
            y1 / original_height,
            x2 / original_width,
            y2 / original_height,
        ]

        # Resize image and mask to 500x500
        #image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_LINEAR)
        #mask = cv2.resize(mask, (500, 500), interpolation=cv2.INTER_NEAREST)

        # Convert to tensors
        image = F.to_tensor(image)
        bbox = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 4]
        label = torch.tensor([annotation['category_id']], dtype=torch.int64)  # Shape: [1]
        mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)  # Shape: [1, H, W]

        # Create target dictionary
        target = {
            "boxes": bbox,
            "labels": label,
            "image_id": torch.tensor([image_id]),
            "masks": mask,
            "area": torch.tensor([annotation['area']], dtype=torch.float32),
            "iscrowd": torch.tensor([annotation.get('iscrowd', 0)], dtype=torch.int64),
        }

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, target

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MaskRCNNDatasetFromCSV(Dataset):
    def __init__(self, csv_path, image_dir, mask_dir, transforms=None, mask_ext=".png"):
        self.data = pd.read_csv(csv_path, header=None)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.mask_ext = mask_ext

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Clean filename
        img_filename = os.path.basename(row[0].strip())
        image_path = os.path.join(self.image_dir, img_filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        image_np = np.array(image)

        if self.transforms:
           image = self.transforms(image=image_np)["image"]
        else:
           image = transforms.ToTensor()(image)

        resized_width, resized_height = 640, 640  # for scaling boxes


        # Bounding box (scale to resized image)
        xmin, xmax, ymin, ymax = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        x1 = int(xmin / original_width * resized_width)
        x2 = int(xmax / original_width * resized_width)
        y1 = int(ymin / original_height * resized_height)
        y2 = int(ymax / original_height * resized_height)

        # Use normalized box for target
        boxes = torch.tensor([[x1 / resized_width, y1 / resized_height,
                               x2 / resized_width, y2 / resized_height]], dtype=torch.float32)

        # Load mask
        mask_path = os.path.join(self.mask_dir, img_filename.replace(".jpg", self.mask_ext))
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = Image.open(mask_path).convert("L").resize((640, 640))
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)
        mask = torch.as_tensor(mask, dtype=torch.uint8).unsqueeze(0)

        area = torch.tensor([(x2 - x1) * (y2 - y1)], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)
        image_id = torch.tensor([idx])
        iscrowd = torch.tensor([0], dtype=torch.int64)

        #image = transforms.ToTensor()(image)
        #if self.transforms:
        #    image = self.transforms(image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "masks": mask
        }

        return image, target

# Recommended normalization transform
normalized_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])




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
from torchvision import transforms
from torchvision.transforms import functional as F

class MaskDataset(Dataset):
    def __init__(self, coco_file, image_folder, mask_folder, transforms=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transforms = transforms

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
        #image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #if self.transforms:
        image = self.transforms(image=image)["image"]
        #else:
        #    image = F.to_tensor(image)
 
        original_height, original_width = image.shape[:2]

        # Load corresponding mask
        mask_filename = f"{image_info['file_name'].split('.')[0]}_{annotation['id']}.png"
        mask_path = os.path.join(self.mask_folder, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Normalize bbox
        x1, y1, w, h = annotation['bbox']
        x2, y2 = x1 + w, y1 + h
        bbox = torch.tensor([
            x1 / original_width,
            y1 / original_height,
            x2 / original_width,
            y2 / original_height,
        ], dtype=torch.float32).unsqueeze(0)

        # Convert image and mask to tensor
        #image = F.to_tensor(image)
        mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)

        #if self.transform:
        #    image = self.transform(image)

        target = {
            "boxes": bbox,
            "labels": torch.tensor([annotation['category_id']], dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "masks": mask,
            "area": torch.tensor([annotation['area']], dtype=torch.float32),
            "iscrowd": torch.tensor([annotation.get('iscrowd', 0)], dtype=torch.int64),
        }

        return image, target

# Recommended normalization transform
#normalized_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                            std=[0.229, 0.224, 0.225])




from torchvision import transforms

#transform_pipeline = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225])
#])

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import DataLoader
import time
from dataset import *

from engine import train_one_epoch, evaluate

# 1️⃣ Build the Model
def build_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


# 2️⃣ Load Model, Optimizer, Scheduler, and Epoch from Checkpoint
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model weights loaded.")

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Optimizer state loaded.")

    if scheduler and 'lr_scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print("✅ Scheduler state loaded.")

    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"✅ Resuming training from epoch {start_epoch}")

    return model, optimizer, scheduler, start_epoch


# 3️⃣ Training Loop
def train(num_classes, dataset, batch_size=4, num_epochs=10, checkpoint_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = build_model(num_classes)
    model.to(device)

    # Define optimizer & learning rate scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Load checkpoint if provided
    if checkpoint_path:
        model, optimizer, scheduler, start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    # DataLoader (Replace this with your dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
   
    
    for epoch in range(num_epochs):
       train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
       scheduler.step()
       #evaluate(model, data_loader_test, device=device)
    torch.save({
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
     }, "checkpoint_ff.pth") 

from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

#csv_path = "../sunlamp_test.csv"
#image_dir = "../sun_lamp/images/"
#mask_folder = "../sunlamp/masks/"
coco_file = "train/_annotations.coco.json"
image_dir = "train/"
mask_folder = "masks_with_debug/"

dataset = MaskDataset(coco_file, image_dir, mask_folder)
#dataset = MaskRCNNDatasetFromCSV(csv_path, image_dir, mask_dir=mask_folder,transforms=transform, mask_ext=".jpg")

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))





train(
    num_classes=2,
    dataset=dataset,
    batch_size=16,
    num_epochs=5,
    checkpoint_path="checkpoint_f.pth"
)




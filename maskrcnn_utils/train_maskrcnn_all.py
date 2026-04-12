import torch 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
#from dataset import *
from dataset_finalversion import *

from torch.utils.data import ConcatDataset, DataLoader


from torch.utils.data import Dataset

class RepeatedDataset(Dataset):
    def __init__(self, base_dataset, repeat=40):
        self.base = base_dataset
        self.repeat = repeat

    def __len__(self):
        return len(self.base) * self.repeat

    def __getitem__(self, idx):
        return self.base[idx % len(self.base)]



import albumentations as A
from albumentations.pytorch import ToTensorV2

'''
image_only_transform = A.Compose([
    A.Rotate(limit=15, border_mode=0, p=0.8),                        # Simulate attitude variation
    A.Affine(scale=(0.8, 1.2), p=0.7),                               # Scale variance
    A.Perspective(scale=(0.05, 0.1), p=0.3),                         # Orbital distance distortion
    A.RandomBrightnessContrast(0.2, 0.2, p=0.6),                     # Solar glare / earthshine
    A.GaussNoise(var_limit=(10, 50), p=0.5),                         # CCD sensor noise
    A.Resize(640, 640),                                              # Fixed resize
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

'''
#transform = image_only_transform

from torchvision import transforms

transform = transforms.Compose([
    #transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



csv_path = "../test.csv"
image_dir = "../lightbox_all/lightbox_images"
mask_folder = "../lightbox_all/masks/"

dataset_lightbox = MaskRCNNDatasetFromCSV(csv_path, image_dir, mask_dir=mask_folder,transforms=transform, mask_ext=".jpg")


#print(len(dataset_lightbox))
#synthetic_csv='../train_synth_fil.csv'
#synthetic_image_dir='../synthetic_all/'
#synthetic_mask_folder='../synthetic_all/masks'


#dataset_synth = MaskRCNNDatasetFromCSV(synthetic_csv, synthetic_image_dir, mask_dir=synthetic_mask_folder,transforms=transform, mask_ext=".jpg")

#print(len(dataset_synth))
csv_path_sunlamp = "../sunlamp_test.csv"
image_dir_sunlamp = "../sunlamp_all/sunlamp_images/images"
mask_folder_sunlamp = "../sunlamp_all/masks/"

coco_file = "../train/_annotations.coco.json"
image_dir_lab = "../train/"
mask_folder_lab = "../masks_satv2/"

dataset_lab = MaskDataset(coco_file, image_dir_lab, mask_folder_lab,transform=transform)
lab_dataset = RepeatedDataset(dataset_lab, repeat=60)
#print(len(lab_dataset))


dataset_sunlamp = MaskRCNNDatasetFromCSV(csv_path_sunlamp, image_dir_sunlamp, mask_dir=mask_folder_sunlamp,transforms=transform, mask_ext=".jpg")
#print(len(dataset_sunlamp))





from torch.utils.data import ConcatDataset, WeightedRandomSampler

combined_dataset = ConcatDataset([lab_dataset, dataset_sunlamp,dataset_lightbox])



import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

#
# Initialize RandomSampler
sampler = RandomSampler(
    combined_dataset,  # or data_source (any iterable)
    replacement=False,  # Sample without replacement (default)
    num_samples=None,  # Use all samples (default)
    generator=torch.Generator().manual_seed(42)  # Optional: torch.Generator for reproducibility
)



dataloader = DataLoader(combined_dataset, batch_size=16,sampler=sampler, collate_fn=lambda x: tuple(zip(*x)))





import torchvision
import torchvision.models.detection.mask_rcnn





def build_model(num_classes):
    #model_path='/content/checkpoint.pth'
    #weights = torch.load(model_path)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


import torch
import torch.optim as optim
from torchvision.ops import misc as misc_nn_ops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 2 # Background + object
model = build_model(num_classes)
model.to(device)




from engine import train_one_epoch, evaluate

# Initialize model and optimizer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2 # Background + object
model = build_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Train for 10 epochs
num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq=10)
    lr_scheduler.step()
    #evaluate(model, data_loader_test, device=device)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "checkpoint_all_silverconverted.pth")



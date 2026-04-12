import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
#from dataset import *
from dataset_final import *
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

from torchvision import transforms

from torchvision import transforms

transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


csv_path = "test.csv"
image_dir = "images"
mask_folder = "lightbox/masks/"

dataset = MaskRCNNDatasetFromCSV(csv_path, image_dir, mask_dir=mask_folder,transforms=transform, mask_ext=".jpg")


dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))






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
}, "checkpoint.pth")

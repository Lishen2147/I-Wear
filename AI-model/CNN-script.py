import os
import utils
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Define the hyperparameters
NUM_EPOCHS = 25
BATCH_SIZE = 24
NUM_WORKERS = os.cpu_count() - 1
DATASETS_PATH = './datasets'
TRAIN_PATH = os.path.join(DATASETS_PATH, './training_set')
TEST_PATH = os.path.join(DATASETS_PATH, './testing_set')
SAVE_RESULTS_PATH = './outputs'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.enabled = False

# Preprocess the image data
train_transforms = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the training and testing datasets
train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=train_transforms, loader=utils.safe_pil_loader)
test_dataset = datasets.ImageFolder(root=TEST_PATH, transform=test_transforms, loader=utils.safe_pil_loader)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Load the pre-trained EfficientNet model
model = torchvision.models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1')
num_classes = len(train_dataset.classes)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5, inplace=True),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model.to(DEVICE)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)

# Train the model
train_losses, train_accuracies = utils.train_model(model, train_loader, criterion, optimizer, DEVICE, NUM_EPOCHS, SAVE_RESULTS_PATH)
utils.plot_and_save_training_history(train_losses, train_accuracies, SAVE_RESULTS_PATH)
utils.test_model(model, test_loader, criterion, DEVICE, SAVE_RESULTS_PATH)
utils.save_model(model, SAVE_RESULTS_PATH)
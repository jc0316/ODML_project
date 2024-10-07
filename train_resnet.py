import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os
import logging
import matplotlib.pyplot as plt
import argparse

# Set up logging
logging.basicConfig(filename='training_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train ResNet50 with specified input size and method.')
parser.add_argument('--input_size', type=int, choices=[64, 200], required=True,
                    help='Input size for the images (64 or 200).')
parser.add_argument('--method', type=str, choices=['center_crop', 'resize', 'original'], required=True,
                    help='Downsampling method to use.')
args = parser.parse_args()

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Device being used: {device}")

# Downsampling Methods
def get_data_transforms(input_size, method):
    if method == "center_crop":
        return transforms.Compose([
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])
    elif method == "resize":
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])
    elif method == "original":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])

# Function to count FLOPs
def compute_flops(model, input_size=(3, 224, 224), batch_size=1):
    total_flops = 0
    hooks = []

    def conv_hook(layer, input, output):
        batch_size, in_channels, in_h, in_w = input[0].shape
        out_channels, out_h, out_w = output.shape[1:]
        kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
        flops_per_instance = kernel_size * in_channels * out_h * out_w * out_channels
        total_flops_per_layer = batch_size * flops_per_instance
        nonlocal total_flops
        total_flops += total_flops_per_layer

    def linear_hook(layer, input, output):
        input_dim = layer.in_features
        output_dim = layer.out_features
        flops_per_instance = 2 * input_dim * output_dim
        total_flops_per_layer = batch_size * flops_per_instance
        nonlocal total_flops
        total_flops += total_flops_per_layer

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_hook))

    dummy_input = torch.randn(batch_size, *input_size).to(device)

    with torch.no_grad():
        model(dummy_input)

    for hook in hooks:
        hook.remove()

    return total_flops

# Function to measure inference latency
def measure_inference_latency(model, dataloader, num_runs=5):
    model.eval()
    latencies = []

    # Run the model multiple times to measure latency
    for run in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                model(inputs)
        end_time = time.time()
        latency = (end_time - start_time) / len(dataloader.dataset)
        latencies.append(latency)
        if run == 0:
            # Discard the first measurement (warmup)
            continue
        logging.info(f"Run {run}/{num_runs}, Latency: {latency:.6f} seconds per image")

    # Calculate average latency excluding the first run
    avg_latency = sum(latencies[1:]) / (num_runs - 1)
    logging.info(f"Average Inference Latency (excluding first run): {avg_latency:.6f} seconds per image")
    return avg_latency

# Plotting function (optional, if needed)
def plot_metrics(x_data, y_data, xlabel, ylabel, title, filename):
    plt.figure()
    plt.plot(x_data, y_data, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")

# ResNet50 model definition
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, 
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, stride, downsample)

    def forward(self, x):
        return self.conv_block(x)

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Prepare datasets and evaluate the model for specified input size and method
input_size = args.input_size
method = args.method

print(f"\nEvaluating for input size {input_size} with {method} method")
logging.info(f"Evaluating for input size {input_size} with {method} method")

# Get data transforms
if method == 'original':
    data_transform = get_data_transforms(input_size=0, method=method)
else:
    data_transform = get_data_transforms(input_size=input_size, method=method)

# Data directories (assumed to be defined)
data_dir = 'data'  # Replace with your data directory
train_dir = os.path.join(data_dir, 'train')
dev_dir = os.path.join(data_dir, 'dev')
test_dir = os.path.join(data_dir, 'test')

image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transform),
    'dev': datasets.ImageFolder(dev_dir, transform=data_transform),
    'test': datasets.ImageFolder(test_dir, transform=data_transform)
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'dev': DataLoader(image_datasets['dev'], batch_size=32, shuffle=False),
    'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False)
}

num_classes = len(image_datasets['train'].classes)

# Initialize the model
model = ResNet50(num_classes=num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5  # Adjust as needed
best_acc = 0.0

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    logging.info(f'Epoch {epoch+1}/{num_epochs}')
    for phase in ['train', 'dev']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Print progress
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(dataloaders[phase])}')

        # Compute metrics at the end of the epoch
        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if phase == 'dev' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

# Load best model weights
model.load_state_dict(best_model_wts)

# Save the final model weights
model_save_path = f'model_weights_{input_size}_{method}.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}")
logging.info(f"Model weights saved to {model_save_path}")

# Evaluate on test set
model.eval()
running_corrects = 0

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

test_acc = running_corrects.double() / len(image_datasets['test'])
print(f'Test Accuracy: {test_acc:.4f}')
logging.info(f'Test Accuracy: {test_acc:.4f}')

# Compute FLOPs and Latency
input_dims = (3, input_size, input_size) if method != 'original' else (3, 200, 200)
flops = compute_flops(model, input_size=input_dims, batch_size=1)
latency = measure_inference_latency(model, dataloaders['test'])
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

logging.info(f"Parameter Count: {param_count}")
logging.info(f"FLOPs: {flops}")
logging.info(f"Inference Latency: {latency:.6f} seconds per image")

# Store results (if you plan to aggregate them later)
results = {
    'input_size': input_size,
    'method': method,
    'flops': flops,
    'latency': latency,
    'accuracy': test_acc.item(),
    'params': param_count
}

print(f"Results: {results}")

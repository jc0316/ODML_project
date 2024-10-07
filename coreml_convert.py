import torch
import coremltools as ct

# Import your custom ResNet50 model definition (as shared above)
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                     stride=stride, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                     stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, 
                                     bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels * 4)
        self.relu = torch.nn.ReLU(inplace=True)
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

class BasicBlock(torch.nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, stride, downsample)

    def forward(self, x):
        return self.conv_block(x)

class ResNet50(torch.nn.Module):
    def __init__(self, num_classes=1000, input_channels=3):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        self.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, 
                                     padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                                kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return torch.nn.Sequential(*layers)

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

# 1. Instantiate your model and load weights
model = ResNet50(num_classes=29, input_channels=3)
model.load_state_dict(torch.load("/ocean/projects/cis220031p/mmisra/ODML/Lab2/model_weights_64_center_crop.pth"))  # Adjust the path to your .pth file

# Set the model to evaluation mode
model.eval()

# 2. Create an example input tensor that matches the expected input size of 200x200
example_input = torch.rand(1, 3, 64, 64)  # Batch size 1, 3 color channels, 200x200 image

# 3. Convert the model to TorchScript
traced_model = torch.jit.trace(model, example_input)

# 4. Convert the TorchScript model to CoreML format
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],  # Specify input shape (1, 3, 200, 200)
)

# 5. Save the CoreML model to a file
mlmodel.save("model_weights_64_center_crop.mlpackage")

print("Model successfully converted and saved as model_weights_64_center_crop.mlpackage")

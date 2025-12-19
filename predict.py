import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Match your training model exactly
class ThumbnailCNN(nn.Module):
    def __init__(self):
        super(ThumbnailCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 720, 1280)
            dummy_output = self.cnn(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ThumbnailCNN().to(device)
model.load_state_dict(torch.load("thumbnail_model.pth", map_location=device))
model.eval()

# Transform same as training
transform = transforms.Compose([
    transforms.Resize((720, 1280)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Prediction function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return "Good Thumbnail" if predicted.item() == 1 else "Bad Thumbnail"

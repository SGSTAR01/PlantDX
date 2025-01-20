import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, jsonify

app = Flask(__name__)

idenfied_classes = ['Pepper,_bell___Bacterial_spot', 'Grape___Esca_(Black_Measles)', 'Cherry_(including_sour)___healthy', 'Tomato___Early_blight', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Septoria_leaf_spot', 'Grape___healthy', 'Blueberry___healthy', 'Pepper,_bell___healthy', 'Peach___healthy', 'Corn_(maize)___healthy', 'Strawberry___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Potato___Early_blight', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Tomato___Target_Spot', 'Apple___Cedar_apple_rust', 'Corn_(maize)___Common_rust_', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Corn_(maize)___Northern_Leaf_Blight', 'Apple___healthy', 'Raspberry___healthy', 'Tomato___Late_blight', 'Grape___Black_rot', 'Tomato___Bacterial_spot', 'Potato___healthy', 'Apple___Apple_scab', 'Potato___Late_blight', 'Tomato___Leaf_Mold', 'Apple___Black_rot', 'Soybean___healthy']
idenfied_classes.sort()
#Architecture for training

# Convolutional block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet14(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        # Initial convolutional layers
        self.conv1 = ConvBlock(in_channels, 64)                # Output: 64 x H x W
        self.conv2 = ConvBlock(64, 128, pool=True)             # Output: 128 x H/4 x W/4

        # Residual block 1
        self.res1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )

        # Additional convolutional layers
        self.conv3 = ConvBlock(128, 256, pool=True)            # Output: 256 x H/16 x W/16
        self.conv4 = ConvBlock(256, 512, pool=True)            # Output: 512 x H/64 x W/64

        # Residual block 2
        self.res2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )

        # New convolutional layer for deeper structure
        self.conv5 = ConvBlock(512, 1024, pool=True)           # Output: 1024 x H/256 x W/256

        # Residual block 3
        self.res3 = nn.Sequential(
            ConvBlock(1024, 1024),
            ConvBlock(1024, 1024)
        )

        # Updated Classifier with Global Average Pooling
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),                      # Global pooling to 1x1 spatial dimensions
            nn.Flatten(),                                      # Flatten features
            nn.Linear(1024, num_diseases)                      # Fully connected layer
        )

    def forward(self, xb):
        out = self.conv1(xb)                                   # Conv1
        out = self.conv2(out)                                  # Conv2
        out = self.res1(out) + out                             # Residual block 1
        out = self.conv3(out)                                  # Conv3
        out = self.conv4(out)                                  # Conv4
        out = self.res2(out) + out                             # Residual block 2
        out = self.conv5(out)                                  # Conv5
        out = self.res3(out) + out                             # Residual block 3
        out = self.classifier(out)                             # Classifier
        return out

model = ResNet14(3,38)
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    
])
@app.route('/predict', methods=['POST'])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    image = request.files["image"]

    try:
        image = Image.open(image)
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            confidence = torch.nn.functional.softmax(output, dim=1)[0] * 100
            identified_disease = idenfied_classes[predicted[0].item()]
            confidence_value = confidence[predicted[0].item()].item()

            result = {
                "identified_disease": identified_disease,
                "confidence": confidence_value
            }

            return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/predict', methods=['GET'])
def predict_get():
    return jsonify({"error": "Please make a POST request to this endpoint with a file upload."}), 400
    
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Plant Disease Detection API!"}), 200

if __name__ == "__main__":
    from waitress import serve
    print("Server is starting on http://127.0.0.1:8080")
    serve(app, host="127.0.0.1", port=8080)

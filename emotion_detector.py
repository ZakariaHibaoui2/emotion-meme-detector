import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, 2, bias=False)
        self.residual_bn = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.sepConv1 = SeparableConv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.relu = nn.ReLU()
        self.sepConv2 = SeparableConv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.maxp = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        res = self.residual_bn(self.residual_conv(x))
        x = self.relu(self.bn1(self.sepConv1(x)))
        x = self.maxp(self.bn2(self.sepConv2(x)))
        return res + x


class EmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(8, momentum=0.99, eps=1e-3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 3, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(8, momentum=0.99, eps=1e-3)
        self.relu2 = nn.ReLU()
        self.module1 = ResidualBlock(8, 16)
        self.module2 = ResidualBlock(16, 32)
        self.module3 = ResidualBlock(32, 64)
        self.module4 = ResidualBlock(64, 128)
        self.last_conv = nn.Conv2d(128, num_classes, 3, padding=1)
        self.avgp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = self.last_conv(x)
        x = self.avgp(x)
        return x.view(x.shape[0], -1)


EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprised", "Neutral"]


class EmotionDetector:
    def __init__(self, model_path="models/emotion_model.t7"):
        self.device = "cpu"
        self.net = EmotionModel(num_classes=7)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(checkpoint["net"])
        self.net.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))

        if len(faces) == 0:
            return None, None

        (x, y, w, h) = faces[0]
        face = gray[y : y + h, x : x + w]
        face = cv2.resize(face, (48, 48))

        img = Image.fromarray(face)
        tensor = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = self.net(tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        emotion = EMOTIONS[pred]
        confidence = probs[0, pred].item()

        return emotion, confidence

import torch
from torchvision.models import inception_v3
from torchvision import transforms
import numpy as np
from PIL import Image
import os


class FIDCalculator:
    def __init__(self, device):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor()
        ])

    def get_features(self, image_paths):
        features = []

        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feat = self.model(img)

            features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def calculate_fid(self, real_paths, fake_paths):
        real_feats = self.get_features(real_paths)
        fake_feats = self.get_features(fake_paths)

        mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
        mu2, sigma2 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

        diff = mu1 - mu2
        fid = diff.dot(diff)

        return fid



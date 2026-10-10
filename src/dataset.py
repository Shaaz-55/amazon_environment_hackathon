import os
import io
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def load_image_with_white_bg(path):
    """
    Loads an image and handles RGBA / palette modes by compositing onto a solid white background.
    """
    with Image.open(path) as img:
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGBA')
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return bg
        else:
            return img.convert('RGB')

class RandomJPEGCompression:
    def __init__(self, quality_range=(50, 90), p=0.3):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            q = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q)
            buf.seek(0)
            return Image.open(buf).convert('RGB')
        return img

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.02, p=0.25):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

def get_train_transforms():
    return transforms.Compose([
        RandomJPEGCompression(quality_range=(50, 90), p=0.3),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.04),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.25),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.02, p=0.25),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class GarbageDataset(Dataset):
    def __init__(self, manifest_path, transform=None):
        with open(manifest_path) as f:
            self.samples = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = load_image_with_white_bg(item["path"])
        label = item["class_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label, item["path"]

if __name__ == "__main__":
    ds = GarbageDataset("manifests/val_manifest.json", transform=get_val_transforms())
    print(f"Validation dataset size: {len(ds)}")
    img, lbl, path = ds[0]
    print(f"Sample 0: tensor shape={img.shape}, label={lbl}, path={path}")

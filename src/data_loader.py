import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Define constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'asl_alphabet_train')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def verify_setup():
    """Verify environment setup and data availability"""
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    required_packages = {
        'torch': torch,
        'PIL': Image,
        'numpy': np,
        'matplotlib': plt
    }
    
    missing_packages = []
    for package, module in required_packages.items():
        if module is None:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(f"Missing required packages: {', '.join(missing_packages)}")

class MinimalPreprocess:
    """Minimal preprocessing preserving original image characteristics"""
    def __init__(self, size=IMG_SIZE):
        self.size = size

    def __call__(self, img):
        try:
            # Keep aspect ratio while resizing
            w, h = img.size
            ratio = min(self.size[0]/w, self.size[1]/h)
            new_size = tuple(int(dim * ratio) for dim in (w, h))
            
            # High quality resize
            img = img.resize(new_size, Image.Resampling.BICUBIC)
            
            # Center with padding
            result = Image.new('RGB', self.size, (255, 255, 255))
            left = (self.size[0] - new_size[0])//2
            top = (self.size[1] - new_size[1])//2
            result.paste(img, (left, top))
            
            return result
        except Exception as e:
            raise RuntimeError(f"Error preprocessing image: {str(e)}")

# Basic transforms
transform = transforms.Compose([
    MinimalPreprocess(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

class ASLDataset(Dataset):
    """ASL Dataset with basic error handling"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if os.path.isfile(img_path):
                        images.append((img_path, self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_path, label = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {str(e)}")

def get_data_loaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    """Create train and validation data loaders with error handling"""
    try:
        verify_setup()
        
        dataset = ASLDataset(data_dir, transform=transform)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # Reduced from 4 for better stability
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    except Exception as e:
        raise RuntimeError(f"Error creating data loaders: {str(e)}")

if __name__ == '__main__':
    try:
        train_loader, val_loader = get_data_loaders()
        print(f"Successfully loaded {len(train_loader)} training batches and {len(val_loader)} validation batches")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
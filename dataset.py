import cv2
import numpy as np

from torch.utils.data import Dataset

class KiumDataset(Dataset):
  def __init__(self, img_paths, labels, img_type, transforms=None):
    self.img_paths = img_paths
    self.labels = labels
    self.img_type = img_type
    self.transforms = transforms
  
  def __getitem__(self, idx):
    img = cv2.imread(f"{self.img_paths[idx]}{self.img_type}.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if self.transforms is not None:
      img = self.transforms(image=img)["image"]
    
    if self.labels is not None:
      return img, self.labels[idx]
    else:
      return img
    
  def __len__(self):
    return len(self.img_paths)
  

class KiumDataset_v1(Dataset):
  def __init__(self, img_paths, labels, transforms=None):
    self.img_paths = img_paths
    self.labels = labels
    self.transforms = transforms
  
  def __getitem__(self, idx):
    imgs = []
    for n in ["LI-A", "LI-B", "LV-A", "LV-B", "RI-A", "RI-B", "RV-A", "RV-B"]:
      img = cv2.imread(f"{self.img_paths[idx]}{n}.jpg")
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
      if self.transforms is not None:
        img = self.transforms(image=img)["image"]
      imgs.append(img)
    
    img = np.concatenate(imgs, axis=0)
  
    if self.labels is not None:
      return img, self.labels[idx]
    else:
      return img
    
  def __len__(self):
    return len(self.img_paths)
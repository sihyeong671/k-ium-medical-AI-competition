import argparse
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import KiumDataset


def load_model(type_name):
  model = torch.load(f"ckpt/densenet121_{type_name}.pth")
  return model

def predict():
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  label_dict = {
    "LI-A": ["L_ACA", "L_MCA", "L_ACOM", "L_PCOM", "L_ICA"],
    "LI-B": ["L_ACA", "L_MCA", "L_AntChor", "L_PCOM", "L_ICA"],
    "LV-A": ["L_PCA", "BA", "L_SCA", "L_VA"],
    "LV-B": ["L_PCA", "BA", "L_PICA", "L_VA"],
    "RI-A": ["R_ACA", "R_MCA", "R_ACOM", "R_PCOM", "R_ICA"],
    "RI-B": ["R_ACA", "R_MCA", "R_AntChor", "R_PCOM", "R_ICA"],
    "RV-A": ["R_PCA", "BA", "R_SCA", "R_VA"],
    "RV-B": ["R_PCA", "BA", "R_PICA", "R_VA"],
  }
  
  # csv 가져오기
  df = pd.read_csv("./data/train_set/train.csv")
  df["Index"] = df["Index"].apply(lambda x: os.path.join("data/train_set", str(x)))
  
  test_transforms = A.Compose([
    A.Normalize(0.5, 0.225),
    A.Resize(256, 256),
    ToTensorV2()
  ])
  
  
  # 각 type 이미지에 대해 알맞게 추론
  result_dict = {}
  for type_name in ["LI-A", "LI-B", "LV-A", "LV-B", "RI-A", "RI-B", "RV-A", "RV-B"]:
    model = load_model(type_name)
    
    dataset = KiumDataset(img_paths=df["Index"].values, labels=None, img_type=type_name, transforms=test_transforms)
    dataloader = DataLoader(
      dataset=dataset,
      batch_size=16,
      shuffle=False
    )
    
    model.to(device)
    model.eval()
    result = None
    with torch.no_grad():
      for imgs in tqdm(dataloader):
        imgs = imgs.to(device)
        
        output = model(imgs)
        
        # sigmoid 적용
        probs = F.sigmoid(output)
        probs = probs.cpu().detach().numpy()
        if result is None:
          result = probs
        else:
          result = np.concatenate([result, probs], axis=0)
    
    result_df = pd.DataFrame(result, columns=[label_dict[type_name]])
    result_df.to_csv(f"./test{type_name}.csv", index=False)
        
  
  # 8개 추론
  
  # csv 파일 병합 및 voting
  
  # csv저장
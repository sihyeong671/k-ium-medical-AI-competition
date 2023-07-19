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

from dataset import KiumDataset, KiumDataset_v1


def load_model():
  model = torch.load(f"ckpt/effnetb4.pth")
  return model

def predict():
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  # label_dict = {
  #   "LI-A": ["L_ACA", "L_MCA", "L_ACOM", "L_PCOM", "L_ICA"],
  #   "LI-B": ["L_ACA", "L_MCA", "L_AntChor", "L_PCOM", "L_ICA"],
  #   "LV-A": ["L_PCA", "BA", "L_SCA", "L_VA"],
  #   "LV-B": ["L_PCA", "BA", "L_PICA", "L_VA"],
  #   "RI-A": ["R_ACA", "R_MCA", "R_ACOM", "R_PCOM", "R_ICA"],
  #   "RI-B": ["R_ACA", "R_MCA", "R_AntChor", "R_PCOM", "R_ICA"],
  #   "RV-A": ["R_PCA", "BA", "R_SCA", "R_VA"],
  #   "RV-B": ["R_PCA", "BA", "R_PICA", "R_VA"],
  # }
  
  # csv 가져오기
  df = pd.read_csv("./test_set/test.csv")
  df["path"] = df["Index"].apply(lambda x: os.path.join("test_set", f"{x:04d}"))
  
  test_transforms = A.Compose([
    A.Normalize(0.5, 0.225),
    A.Resize(256, 256),
    ToTensorV2()
  ])
  
  model = load_model()
  
  dataset = KiumDataset_v1(img_paths=df["path"].values, labels=None, transforms=test_transforms)
  dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
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
        
  columns = ["Index","Aneurysm","L_ICA","R_ICA","L_PCOM","R_PCOM","L_AntChor","R_AntChor","L_ACA","R_ACA","L_ACOM","R_ACOM","L_MCA","R_MCA","L_VA","R_VA","L_PICA","R_PICA","L_SCA","R_SCA","BA","L_PCA","R_PCA"]
  result_df = pd.DataFrame(columns=columns)
  empty = [0] * len(df["Aneurysm"])
  for c in result_df.columns:
    if c == "Index":
      result_df[c] = df["Index"].values
    elif c == "Aneurysm":
      result_df[c] = result
    else:
      result_df[c] = empty
  result_df.to_csv(f"./output.csv", index=False)
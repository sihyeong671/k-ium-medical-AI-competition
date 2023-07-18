import os
import random
from copy import deepcopy
import logging

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

import timm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dataset import KiumDataset, KiumDataset_v1
from model import Densenet121, Densenet121_v1, Effnetb4_v1
from loss import W_BCEWithLogitsLoss

def seed_everything(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True


class Trainer:
  def __init__(self):
    self.epoch = 20
    self.lr = 0.001
    self.seed = 44
    self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    self.root_dir = "./data"
    self.img_type = "RV-B"
    self.label_dict = {
      "LI-A": ["Aneurysm", "L_ACA", "L_MCA", "L_ACOM", "L_PCOM", "L_ICA"],
      "LI-B": ["Aneurysm", "L_ACA", "L_MCA", "L_AntChor", "L_PCOM", "L_ICA"],
      "LV-A": ["Aneurysm", "L_PCA", "BA", "L_SCA", "L_VA"],
      "LV-B": ["Aneurysm", "L_PCA", "BA", "L_PICA", "L_VA"],
      "RI-A": ["Aneurysm", "R_ACA", "R_MCA", "R_ACOM", "R_PCOM", "R_ICA"],
      "RI-B": ["Aneurysm", "R_ACA", "R_MCA", "R_AntChor", "R_PCOM", "R_ICA"],
      "RV-A": ["Aneurysm", "R_PCA", "BA", "R_SCA", "R_VA"],
      "RV-B": ["Aneurysm", "R_PCA", "BA", "R_PICA", "R_VA"],
    }
    
    # LOGGER
    self.logger = logging.getLogger()
    self.logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    self.logger.addHandler(stream_handler)
  
  def setup(self):
    
    self.model = Effnetb4_v1()
    self.loss_fn = nn.BCELoss()
    
    #  랜덤 시드 설정
    seed_everything(self.seed)
    
    # df 수정
    self.df = pd.read_csv(os.path.join(self.root_dir, "train_set/train.csv"))
    self.df["path"] = self.df["Index"].apply(lambda x: os.path.join(self.root_dir, "train_set", str(x)))
    # tmp_df = pd.read_csv(os.path.join(self.root_dir, "train_set/train.csv"))
    # self.df = pd.DataFrame(columns=["path", *self.label_dict[self.img_type]])
    
    # paths = []
    # labels = {l: [] for l in self.label_dict[self.img_type]}
    
    # for idx, row in tmp_df.iterrows():
    #   paths.append(os.path.join(self.root_dir, "train_set", str(row["Index"])))
      
    #   for label in labels:
    #     labels[label].append(1 if row[label] == 1 else 0)
    
    # self.df["path"] = paths
    # for k, v in labels.items():
    #   self.df[k] = v
    
    x_train, x_val, y_train, y_val = train_test_split(
      self.df["path"].values,
      self.df["Aneurysm"].values,
      test_size=0.2,
      random_state=self.seed,
      stratify=self.df["Aneurysm"])
    
    # 데이터 셋 정의
    train_transforms = A.Compose([
      A.Normalize(0.5, 0.225),
      A.GaussianBlur(),
      A.ElasticTransform(alpha=100, sigma=10),
      A.Resize(256, 256),
      ToTensorV2()
    ])
    
    val_transforms = A.Compose([
      A.Normalize(0.5, 0.225),
      A.Resize(256, 256),
      ToTensorV2()
    ])
    
    train_dataset = KiumDataset_v1(img_paths=x_train, labels=y_train, transforms=train_transforms)
    val_dataset = KiumDataset_v1(img_paths=x_val, labels=y_val, transforms=val_transforms)
    
    self.train_dataloader = DataLoader(
      dataset=train_dataset,
      batch_size=32, 
      shuffle=True
    )
    
    self.val_dataloader = DataLoader(
      dataset=val_dataset,
      batch_size=32,
      shuffle=False
    )
    
      
  def valid(self):
    """
    Return:
      (val_loss, val_acc)
    """
    self.model.eval()
    val_loss = []
    val_acc = []
    with torch.no_grad():
      for imgs, labels in tqdm(self.val_dataloader):
        imgs = imgs.to(self.device)
        labels = labels.float().to(self.device)

        probs = self.model(imgs)
        probs = probs.squeeze(-1)
        loss = self.loss_fn(probs, labels)
        probs = probs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        
        preds = probs > 0.5
        batch_acc = (labels == preds).mean()
        val_acc.append(batch_acc)
        val_loss.append(loss.item())
        
    return np.mean(val_loss), np.mean(val_acc)
  
  def train(self):
    self.model.to(self.device)
    self.loss_fn.to(self.device)
    
    optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer=optimizer,
      mode='max',
      factor=0.5,
      patience=3,
      cooldown=5,
      min_lr=1e-9,
      threshold_mode='abs',
    )

    best_val_acc = 0
    best_model = None
    
    for epoch in range(1, self.epoch+1):
      self.model.train()
      train_loss_lst = []
      for imgs, labels in tqdm(self.train_dataloader):
        imgs = imgs.to(self.device)
        labels = labels.float().to(self.device)

        optimizer.zero_grad()
        output = self.model(imgs)
        output = output.squeeze(-1)
        loss = self.loss_fn(output, labels)
        loss.backward()
        
        optimizer.step()
        
        train_loss_lst.append(loss.item())
        
      val_loss, val_acc = self.valid()
      train_loss = np.mean(train_loss_lst)
      self.logger.info(f"EPOCH: {epoch}, TRAIN LOSS: {train_loss:.4f}, VAL LOSS: {val_loss:.4f}, VAL ACC: {val_acc:.4f}")
    
      if lr_scheduler is not None:
        lr_scheduler.step(val_acc)
        
      if best_val_acc <= val_acc:
        best_val_acc = val_acc
        best_model = deepcopy(self.model)
        early_stop = 0
      else:
        early_stop += 1
      
      if early_stop > 5:
        break
      
    torch.save(best_model, f'./ckpt/effnetb4.pth')
    
    
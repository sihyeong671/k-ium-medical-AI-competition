import torch
import torch.nn as nn

class W_BCEWithLogitsLoss(nn.Module):
    
  def __init__(self, w_p = None, w_n = None):
    super().__init__()
    
    self.w_p = w_p
    self.w_n = w_n
      
  def forward(self, logits, labels, epsilon = 1e-7):
      
    ps = torch.sigmoid(logits.squeeze()) 
    
    loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(ps + epsilon))
    loss_neg = -1 * torch.mean(self.w_n * (1-labels) * torch.log((1-ps) + epsilon))
    
    loss = loss_pos + loss_neg
    
    return loss
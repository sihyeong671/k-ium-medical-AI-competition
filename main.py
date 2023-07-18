import argparse

from trainer import Trainer
from predictor import predict

def train():
  trainer = Trainer()
  trainer.setup()
  trainer.train()

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default="train")
  args = parser.parse_args()
  
  if args.mode == "train":
    train()
  elif args.mode == "test":
    predict()
from trainer import Trainer
from predictor import predict

def train():
  trainer = Trainer()
  trainer.setup()
  trainer.train()

if __name__ == "__main__":
  mode = "train"
  if mode == "train":
    train()
  elif mode == "test":
    predict()
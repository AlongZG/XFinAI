from torch.utils.tensorboard import SummaryWriter
import sys
import os

sys.path.append('../')
import xfinai_config
from utils import base_io, path_wrapper
from model_layer.model_hub import RNN
from model_layer.model_trainer import RecurrentModelTrainer


def main():
    model_class = RNN
    future_index = 'IC'
    log_dir = f"./model_structure"
    train_model = RecurrentModelTrainer(model_class=model_class, future_index=future_index)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, train_model.model_name))

    for batch_x, _ in train_model.train_loader:
        writer.add_graph(train_model.model, batch_x)
        break


if __name__ == '__main__':
    main()

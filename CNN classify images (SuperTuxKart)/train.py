from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import numpy as np
import torch.utils.tensorboard as tb
import torch.nn.functional as F


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    train_data = load_data("data/train")
    vaild_data = load_data("data/valid")

    n_epochs = 60
    global_step = 0

    for epoch in range(n_epochs):
      train_acc = []
      valid_acc = []
      for i, data in enumerate(train_data):
        # data
        inputs, labels = data
        indexes = torch.randperm(inputs.shape[0])
        inputs = inputs[indexes]
        labels = labels[indexes]

        # define optimizer
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=0.0005)

        # training
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        train_acc.append(accuracy(outputs, labels))
        if i % 100 == 0:
          print("Epoch: ", epoch, " Step:", i, " Loss:", loss)

        train_logger.add_scalar('loss', loss, global_step=global_step)
        global_step += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      train_logger.add_scalar('accuracy', np.mean(train_acc), global_step=global_step)

      for i, data in enumerate(vaild_data,0):
        # data
        inputs, labels = data
        indexes = torch.randperm(inputs.shape[0])
        inputs = inputs[indexes]
        labels = labels[indexes]

        # training
        outputs = model(inputs)
        valid_acc.append(accuracy(outputs, labels))
  
      valid_logger.add_scalar('accuracy', np.mean(valid_acc), global_step=global_step)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)

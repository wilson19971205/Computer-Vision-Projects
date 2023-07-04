import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms as T
import torch.utils.tensorboard as tb
import torch.nn.functional as F

def train(args):
    from os import path
    model = FCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    loss_func.to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    epochs = 50

    train_trans = T.Compose((T.RandomCrop(96), T.RandomHorizontalFlip(), T.ColorJitter(0.5, 0.5, 0.5, 0.5), T.ToTensor()))

    train_data = load_dense_data('dense_data/train', transform=train_trans)
    train_val = load_dense_data('dense_data/valid')

    for epoch in range(epochs):
        model.train()
        count = 0
        total_loss = 0
        accuracy = 0
        for data, label in train_data:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss = loss_func(pred, label.long())
            total_loss = total_loss + loss.item()
            accuracy += (pred.argmax(1) == label).float().mean().item()
            count += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch: " + str(epoch) + ", Loss: " + str(total_loss/count))
        scheduler.step(accuracy/count)

        model.eval()
        count = 0
        accuracy = 0
        for data, label in train_val:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            accuracy += (pred.argmax(1) == label).float().mean().item()
            count += 1
        print("Epoch: " + str(epoch) + ", Accuracy: " + str(accuracy/count))
        if accuracy/count > 0.845:
            print("break")
            break

    

    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(T.label_to_pil_image(lbls[0].cpu()).
                                       convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(T.
                                            label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                            convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)

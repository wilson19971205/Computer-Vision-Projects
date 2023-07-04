from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torchvision.transforms as T
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    loss_func.to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    epochs = 50

    train_trans = T.Compose((T.RandomCrop(32), T.RandomHorizontalFlip(), T.ColorJitter(0.5, 0.5), T.ToPILImage(), T.ToTensor()))

    train_data = load_data('data/train', transform=train_trans)
    train_val = load_data('data/valid')

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
        if accuracy/count > 0.9:
            print("break")
            break

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
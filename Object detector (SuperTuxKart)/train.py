import torch
import torch.nn.functional as F
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = Detector()

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Detector().to(device)
    
    #if args.continue_training:
    #    model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    loss = torch.nn.BCEWithLogitsLoss(reduction='mean').to(device)

    transform = dense_transforms.Compose([dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1), dense_transforms.RandomHorizontalFlip(), dense_transforms.ToTensor(), dense_transforms.ToHeatmap()])
    train_data = load_detection_data('dense_data/train', transform=transform, num_workers=4)

    global_step = 0
    for epoch in range(args.num_epoch):
      print("Epoch:", epoch)
      model.train()
      for image, heatmap, _ in train_data:
        image, heatmap = image.to(device), heatmap.to(device)
        pred = model(image)

        loss = F.binary_cross_entropy_with_logits(pred, heatmap, reduction="mean")

        if global_step % 100 == 0:
          print("loss:", loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if train_logger is not None:
          log(train_logger, image, heatmap, pred, global_step)
        global_step += 1
      model.eval()
    
    save_model(model)
    #raise NotImplementedError('train')


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', '--logger', default='log')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=30)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005)
    parser.add_argument('-c', '--continue_training', action='store_true')
    #parser.add_argument('-t', '--transform', default='Compose(ToTensor(), ToHeatmap()])')
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap()])')

    args = parser.parse_args()
    train(args)

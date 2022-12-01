import os
import time
import torch
import logging
import torch.optim as optim
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from test import eval_given_model

# import models
from models.ResNet import ResNet
from models.DeepSense import DeepSense
from models.Transformer import Transformer

# train utils
from train_utils.pretrain_classifier import pretrain_classifier

# utils
from torch.utils.tensorboard import SummaryWriter
from params.train_params import parse_train_params
from input_utils.multi_modal_dataloader import create_dataloader


def train(args):
    """The specific function for training."""
    # Init data loaders
    train_dataloader, triplet_flag = create_dataloader("train", args, batch_size=args.batch_size, workers=args.workers)
    val_dataloader, _ = create_dataloader("val", args, batch_size=args.batch_size, workers=args.workers)
    test_dataloader, _ = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)
    num_batches = len(train_dataloader)

    # Init the classifier model
    if args.model == "DeepSense":
        classifier = DeepSense(args, self_attention=False)
    elif args.model == "SADeepSense":
        classifier = DeepSense(args, self_attention=True)
    elif args.model == "Transformer":
        classifier = Transformer(args)
    elif args.model == "ResNet":
        classifier = ResNet(args)
    else:
        raise Exception(f"Invalid model provided: {args.model}")
    classifier = classifier.to(args.device)

    # Init the Tensorboard summary writer
    tb_writer = SummaryWriter(args.tensorboard_log)

    # define the loss function
    if args.multi_class:
        classifier_loss_func = nn.BCELoss()
    else:
        classifier_loss_func = nn.CrossEntropyLoss()

    pretrain_classifier(
        args,
        classifier,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        classifier_loss_func,
        tb_writer,
        num_batches,
    )


def main_train():
    """The main function of training"""
    args = parse_train_params()
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.FileHandler(args.train_log_file), logging.StreamHandler()]
    )
    train(args)


if __name__ == "__main__":
    main_train()

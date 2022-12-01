import torch.nn as nn

# import models
from models.ResNet import ResNet
from models.DeepSense import DeepSense
from models.Transformer import Transformer

# utils
from general_utils.time_utils import time_sync
from general_utils.weight_utils import load_model_weight
from params.test_params import parse_test_params
from input_utils.multi_modal_dataloader import create_dataloader
from train_utils.eval_functions import eval_given_model


def test(args):
    """The main function for test."""
    # Init data loaders
    train_dataloader, _ = create_dataloader("train", args, batch_size=args.batch_size, workers=args.workers)
    val_dataloader, _ = create_dataloader("val", args, batch_size=args.batch_size, workers=args.workers)
    test_dataloader, _ = create_dataloader("test", args, batch_size=args.batch_size, workers=args.workers)

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
    classifier = load_model_weight(classifier, args.classifier_weight)

    # define the loss function
    if args.multi_class:
        classifier_loss_func = nn.BCELoss()
    else:
        classifier_loss_func = nn.CrossEntropyLoss()

    # print model layers
    if args.verbose:
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                print(name)

    # Test the loss, acc, and F1 for train/val/test dataset
    train_classifier_loss, train_f1, train_acc, train_precision, train_recall, train_conf_matrix = eval_given_model(
        args,
        classifier,
        train_dataloader,
        classifier_loss_func,
    )
    print(f"Train classifier loss: {train_classifier_loss: .5f}")
    print(f"Train acc: {train_acc: .5f}, train precision: {train_precision: .5f}, train recall: {train_recall: .5f}, train f1: {train_f1: .5f}")
    print(f"[{train_acc: .5f}, {train_precision: .5f}, {train_recall: .5f}, {train_f1: .5f}]")
    print(f"Train confusion matrix:\n {train_conf_matrix}")
    if args.eval_detector:
        eval_miss_detector("Train", miss_simulator)

    val_classifier_loss, val_f1, val_acc, val_precision, val_recall, val_conf_matrix = eval_given_model(
        args,
        classifier,
        val_dataloader,
        classifier_loss_func,
    )
    print(f"Val classifier loss: {val_classifier_loss: .5f}")
    print(f"Val acc: {val_acc: .5f}, val precision: {val_precision: .5f}, val recall: {val_recall: .5f}, val f1: {val_f1: .5f}")
    print(f"[{val_acc: .5f}, {val_precision: .5f}, {val_recall: .5f}, {val_f1: .5f}]")
    print(f"Val confusion matrix:\n {val_conf_matrix}")
    if args.eval_detector:
        eval_miss_detector("Val", miss_simulator)

    test_classifier_loss, test_f1, test_acc, test_precision, test_recall, test_conf_matrix = eval_given_model(
        args, classifier, test_dataloader, classifier_loss_func
    )
    
    print(f"Test classifier loss: {test_classifier_loss: .5f}")
    print(f"Test acc: {test_acc: .5f}, test precision: {test_precision: .5f}, test recall: {test_recall: .5f}, test f1: {test_f1: .5f}")
    print(f"[{test_acc: .5f}, {test_precision: .5f}, {test_recall: .5f}, {test_f1: .5f}]")

    print(f"Test confusion matrix:\n {test_conf_matrix}")

def main_test():
    """The main function of training"""
    args = parse_test_params()
    test(args)


if __name__ == "__main__":
    main_test()

from ast import arg
import os
import argparse
import numpy as np

from params.output_paths import set_model_weight_file, set_output_paths, set_model_weight_folder
from params.params_util import *
from input_utils.yaml_utils import load_yaml


def parse_args(option="train"):
    """
    Parse the args.
    """
    parser = argparse.ArgumentParser()

    # dataset config
    parser.add_argument(
        "-dataset",
        type=str,
        default="RealWorld_HAR",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "-model",
        type=str,
        default="DeepSense",
        help="The backbone classification model to use.",
    )

    # training and inference mode
    parser.add_argument(
        "-train_mode",
        type=str,
        default="noisy",
        help="The used mode for model training (original/separate/random/noisy).",
    )
    parser.add_argument(
        "-inference_mode",
        type=str,
        default="noisy",
        help="The used mode for model inference (original/separate/random/noisy).",
    )
    parser.add_argument(
        "-stage",
        type=str,
        default="pretrain_handler",
        help="The train/inference stage for random/noisy modes, pretrain_classifier/pretrain_handler/finetune.",
    )

    # model, miss generator, tracker and handler configs
    parser.add_argument(
        "-miss_modalities",
        type=str,
        default=None,
        help="Used in inference and train-separated, providing the missing modalities separated by ,",
    )
    parser.add_argument(
        "-miss_detector",
        type=str,
        default="FakeDetector",
        help="The approach used to detect the noise.",
    )
    parser.add_argument(
        "-miss_handler",
        type=str,
        default="FakeHandler",
        help="The approach used to handle the missing modalities.",
    )

    # related to noise generator
    parser.add_argument(
        "-noise_std",
        type=float,
        default=1,
        help="The standard deviation of the added noise in the noisy generator",
    )
    parser.add_argument(
        "-noise_mode",
        type=str,
        default="fixed",
        help="The standard deviation of the added noise in the noisy generator",
    )

    # weight path
    parser.add_argument(
        "-model_weight",
        type=str,
        default=None,
        help="Specify the model weight path to evaluate.",
    )

    # hardware config
    parser.add_argument(
        "-batch_size",
        type=int,
        default=64,
        help="Specify the batch size for training.",
    )
    parser.add_argument(
        "-gpu",
        type=int,
        default=None,
        help="Specify which GPU to use.",
    )

    # specify whether to show detailed logs
    parser.add_argument(
        "-verbose",
        type=str,
        default="false",
        help="Whether to show detailed logs.",
    )

    # training configurations
    parser.add_argument(
        "-lr",
        type=float,
        default=None,
        help="Specify the learning rate to try.",
    )

    # evaluation configurations
    parser.add_argument(
        "-eval_detector",
        type=str,
        default="false",
        help="Whether to evaluate the noise detector",
    )
    parser.add_argument(
        "-save_emb",
        type=str,
        default="false",
        help="Whether to save the encoded embeddings.",
    )

    args = parser.parse_args()

    # set option first
    args.option = option

    # gpu configuration
    if args.gpu is None:
        args.gpu = 0
    args.device = select_device(str(args.gpu))
    args.half = False  # half precision only supported on CUDA

    # retrieve the user name
    args.username = get_username()

    # parse the model yaml file
    dataset_yaml = f"./data/{args.dataset}.yaml"
    args.dataset_config = load_yaml(dataset_yaml)

    # verbose
    args.verbose = str_to_bool(args.verbose)
    args.eval_detector = str_to_bool(args.eval_detector)
    args.save_emb = str_to_bool(args.save_emb)

    # threshold
    args.threshold = 0.5

    # dataloader config
    args.workers = 10

    # triplet batch size
    # args.triplet_batch_size = int(args.batch_size / 3)

    # Sing-class problem or multi-class problem
    if args.dataset in {}:
        args.multi_class = True
    else:
        args.multi_class = False

    # process the missing modalities,
    if args.miss_modalities is not None:
        args.miss_modalities = set(args.miss_modalities.split(","))
        print(f"Missing modalities: {args.miss_modalities}")
    else:
        args.miss_modalities = set()

    # automatically set the miss generator and the detector
    args = auto_select_miss_generator(args)
    args = auto_select_miss_detector(args)
    args = auto_select_miss_handler(args)

    # set output path
    args = set_model_weight_folder(args)
    args = set_model_weight_file(args)
    args = set_output_paths(args)

    # perform sanity check on the configuration
    args = sanity_check(args)

    return args

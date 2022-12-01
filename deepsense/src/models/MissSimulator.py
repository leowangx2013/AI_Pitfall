import json
import os
import torch
import logging
import torch.nn as nn
import numpy as np

# miss generator
from miss_generator.separate_miss_generator import SeparateMissGenerator
from miss_generator.random_miss_generator import RandomMissGenerator
from miss_generator.noisy_generator import NoisyGenerator
from miss_generator.no_generator import NoGenerator

# miss detectors (more to add)
from miss_detector.fake_detector import FakeDetector
from miss_detector.gt_detector import GtDetector
from miss_detector.density_detector import DensityDetector
from miss_detector.recon_detector import ReconstructionDetector
from miss_detector.vae_detector import VAEDetector
from miss_detector.vae_plus_detector import VAEPlusDetector

# miss handlers (more to add)
from miss_handler.fake_handler import FakeHandler
from miss_handler.separate_handler import SeparateHandler
from miss_handler.cascaded_residual_autoencoder import CascadedResidualAutoencoder
from miss_handler.cyclic_autoencoder import CyclicAutoencoder
from miss_handler.gate_handler import GateHandler
from miss_handler.ccm_handler import CcmHandler
from miss_handler.matrix_completion_handler import MatrixCompletionHandler
from miss_handler.adversarial_autoencoder import AdAutoEncoder

from miss_handler.resilient_handler import ResilientHandler

# from miss_handler.new_resilient_handler import ResilientHandler

# tensor utils
from general_utils.tensor_utils import bisc_to_bcis, bcis_to_bisc
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

from models.VAE import VAE

"""
Three levels of experiment:
    1) Original: Test the backbone network performance.
    2) Random: Test the model performance on part of the modalities. The missing modalities are known in advance.
    3) Noisy: Detect the noisy/missing modalities first, and then handle these modalities in the handler.

Three componenets in the simulator:
    1) Miss generator:
        * MissingModalityGenerator: Randomly masking some of the modalities with 0s.
        * NoisyModalityGenerator: Randomly add Gaussian noises to some of the modalities.
    2) Miss detector:
        * FakeDetector: Directly use the information from the generator to detect the missing modalities.
    3) Miss handler:
        * Process the data with missing modalities.
        * Output the loss associated with the handler module.
"""


class MissSimulator(nn.Module):
    def __init__(self, args) -> None:
        """This function is used to setup the miss generator, detector, and the handler.

        Args:
            model (_type_): _description_
        """
        super().__init__()
        self.args = args
        self.mode = self.args.train_mode if self.args.option == "train" else self.args.inference_mode
        print(f"[Option]: {args.option}, mode: {self.mode}, stage: {args.stage}")
        print(
            f"[Miss simulator config]: generator: {args.miss_generator}-{args.noise_mode}, detector: {args.miss_detector}, handler: {args.miss_handler}"
        )

        # Step 1.1: Init the missing modality generator
        if args.miss_generator == "SeparateGenerator":
            self.miss_generator = SeparateMissGenerator(args)
        elif args.miss_generator == "RandomGenerator":
            self.miss_generator = RandomMissGenerator(args)
        elif args.miss_generator == "NoisyGenerator":
            self.miss_generator = NoisyGenerator(args)
        elif args.miss_generator == "NoGenerator":
            self.miss_generator = NoGenerator(args)
        else:
            raise Exception(f"Invalid missing modality **generator** provided: {args.miss_generator}")

        # Step 1.2: Init the missing modality detector
        if args.miss_detector == "FakeDetector":
            self.miss_detector = FakeDetector(args)
        elif args.miss_detector == "GtDetector":
            self.miss_detector = GtDetector(args)
        elif args.miss_detector == "DensityDetector":
            self.miss_detector = DensityDetector(
                args,
                in_channels=args.dataset_config[args.model]["loc_mod_out_channels"],
            )
        elif args.miss_detector == "ReconstructionDetector":
            self.miss_detector = ReconstructionDetector(
                args,
                in_channels=args.dataset_config[args.model]["loc_mod_out_channels"],
            )
        elif args.miss_detector == "VAEDetector":
            self.miss_detector = VAEDetector(
                args,
                in_channels=args.dataset_config[args.model]["loc_mod_out_channels"],
            )
        elif args.miss_detector == "VAEPlusDetector":
            self.miss_detector = VAEPlusDetector(
                args,
                in_channels=args.dataset_config[args.model]["loc_mod_out_channels"],
            )
        else:
            raise Exception(f"Invalid missing modality **detector** provided: {args.miss_detector}")

        # Step 1.3: Init the missing modality handler
        if args.miss_handler == "FakeHandler":
            self.miss_handler = FakeHandler(args)
        elif args.miss_handler == "SeparateHandler":
            self.miss_handler = SeparateHandler(args)
        elif args.miss_handler == "CascadedResidualAutoencoder":
            self.miss_handler = CascadedResidualAutoencoder(
                args,
                in_channels=args.dataset_config[args.model]["loc_mod_out_channels"],
            )
        elif args.miss_handler == "CyclicAutoencoder":
            self.miss_handler = CyclicAutoencoder(
                args,
                in_channels=args.dataset_config[args.model]["loc_mod_out_channels"],
            )
        elif args.miss_handler == "GateHandler":
            self.miss_handler = GateHandler(args)
        elif args.miss_handler == "CcmHandler":
            self.miss_handler = CcmHandler(args)
        elif args.miss_handler == "MatrixCompletionHandler":
            self.miss_handler = MatrixCompletionHandler(args)
        elif args.miss_handler == "AdAutoencoder":
            self.miss_handler = AdAutoEncoder(
                args,
                in_channels=args.dataset_config[args.model]["loc_mod_out_channels"],
            )
        elif args.miss_handler == "ResilientHandler":
            self.miss_handler = ResilientHandler(args)
        else:
            raise Exception(f"Invalid missing modality **handler** provided: {args.miss_handler}")

        # store the dt_loc_miss_masks and gt_loc_miss_masks for evaluate detection accuracy
        self.dt_flatten_miss_mask_list = []
        self.dt_flatten_loss_list = []
        self.gt_flatten_miss_mask_list = []
        self.gt_mod_miss_mask_list = []

    def forward(self, loc_mod_features, BISC_order=False):
        """Process the loc_mod_features in miss generator/detector/handler

        Args:
            loc_mod_features (_type_): {loc: [b, c, i, s]}
            BISC_order: only True for the Transformer model.
        Return:
            out_loc_mod_features: the returned loc_mod_features, either reconstructed or un-reconstructed.
            dt_loc_miss_ids: the detected missing modality IDs for each location.
            miss_handler_loss: only used in training, counting the loss related to the handler.
        Note:
            1) args.train_mode and args.train_stage decides what network to test
            2) args.inference_mode decides how to test
        """
        # Step 2.1: Unify the shape to [b, c, i, s] at the input
        if BISC_order:
            in_loc_mod_features = bisc_to_bcis(loc_mod_features)
        else:
            in_loc_mod_features = loc_mod_features

        # Step 2.2: Generate missing modalities
        dirty_loc_mod_features = dict()
        gt_loc_miss_masks = dict()
        for loc in in_loc_mod_features:
            loc_masked_features, loc_miss_masks = self.miss_generator(loc, in_loc_mod_features[loc])
            dirty_loc_mod_features[loc] = loc_masked_features
            gt_loc_miss_masks[loc] = loc_miss_masks

        # Step 2.3: Detect missing modalities, the handlers either consume dt_loc_miss_masks or dt_loc_losses
        dt_loc_miss_masks, dt_loc_losses = self.miss_detector(dirty_loc_mod_features, gt_loc_miss_masks)

        # Step 2.4: Clean up the missing modalties features if required, used for training AE handlers.
        if self.args.miss_generator == "NoisyGenerator" and self.miss_handler.clean_flag and self.miss_handler.training:
            dirty_loc_mod_features = self.clean_noisy_data(dirty_loc_mod_features, dt_loc_miss_masks)

        # Step 2.5: Buffer the dt_loc_miss_masks and gt_loc_miss_mask
        self.save_loc_miss_masks(dt_loc_miss_masks, gt_loc_miss_masks, dt_loc_losses)

        # Step 2.6: Handle missing modalities and calculate the handler loss
        recon_loc_mod_features = dict()
        for loc in dirty_loc_mod_features:
            recon_loc_mod_features[loc] = self.miss_handler(
                dirty_loc_mod_features[loc],
                dt_loc_miss_masks[loc],
                dt_loc_losses[loc],
            )

        # Step 2.7: Convert back to original shape at the output
        if BISC_order:
            recon_loc_mod_features = bcis_to_bisc(recon_loc_mod_features)
            dt_loc_miss_masks = bcis_to_bisc(dt_loc_miss_masks)

        return recon_loc_mod_features, dt_loc_miss_masks

    def clean_noisy_data(self, loc_mod_features, dt_loc_miss_masks):
        """Convert the detected missing modalities to 0."""
        for loc in loc_mod_features:
            loc_mod_features[loc] = loc_mod_features[loc] * dt_loc_miss_masks[loc]

        return loc_mod_features

    def save_loc_miss_masks(self, dt_loc_miss_masks, gt_loc_miss_masks, dt_loc_losses):
        """Save the DT and GT loc miss masks for future miss detection evaluation, in [b * s] shape"""
        for loc in gt_loc_miss_masks:
            tmp_dt_masks = dt_loc_miss_masks[loc].detach()
            tmp_gt_masks = gt_loc_miss_masks[loc].detach()

            tmp_dt_masks = tmp_dt_masks.permute([0, 3, 1, 2]).mean(dim=[2, 3])
            tmp_gt_masks = tmp_gt_masks.permute([0, 3, 1, 2]).mean(dim=[2, 3])

            self.gt_mod_miss_mask_list.append(tmp_gt_masks.cpu().numpy())

            tmp_dt_masks = list(tmp_dt_masks.flatten().cpu().numpy())
            tmp_gt_masks = list(tmp_gt_masks.flatten().cpu().numpy())

            if self.args.miss_detector in {"VAEPlusDetector", "VAEDetector", "ReconstructionDetector"}:
                tmp_dt_losses = list(-dt_loc_losses[loc].detach().flatten().cpu().numpy())
            elif self.args.miss_detector in {"FakeDetector", "GtDetector"}:
                tmp_dt_losses = [0]
            else:
                tmp_dt_losses = list(dt_loc_losses[loc].detach().flatten().cpu().numpy())

            self.dt_flatten_miss_mask_list.extend(tmp_dt_masks)
            self.gt_flatten_miss_mask_list.extend(tmp_gt_masks)
            self.dt_flatten_loss_list.extend(tmp_dt_losses)

    def eval_miss_detector(self, eval_roc=False, eval_auc_score=False, save_emb=False):
        """Evalaute the performance of miss detector."""
        det_acc = accuracy_score(self.gt_flatten_miss_mask_list, self.dt_flatten_miss_mask_list)
        det_f1 = f1_score(self.gt_flatten_miss_mask_list, self.dt_flatten_miss_mask_list)
        det_conf_matrix = confusion_matrix(self.gt_flatten_miss_mask_list, self.dt_flatten_miss_mask_list)

        # AUC score
        if eval_auc_score:
            det_auc = roc_auc_score(self.gt_flatten_miss_mask_list, self.dt_flatten_loss_list)
        else:
            det_auc = -1

        # evaluate the ROC curve
        if eval_roc:
            self.eval_roc_curve()

        # save the latent embeddings
        if self.args.miss_detector in {"VAEPlusDetector", "ReconstructionDetector"} and save_emb:
            self.save_emb()

        # reset buffer after the evaluation
        self.gt_flatten_miss_mask_list = []
        self.dt_flatten_miss_mask_list = []
        self.dt_flatten_loss_list = []

        return det_acc, det_f1, det_conf_matrix, det_auc

    def eval_roc_curve(self):
        """Evaluate the ROC curve according to the detection losses and the labels."""
        fpr, tpr, thresholds = roc_curve(self.gt_flatten_miss_mask_list, self.dt_flatten_loss_list)

        # save into the log
        args = self.args

        log_file = os.path.join(
            args.log_path,
            f"{args.miss_detector}_{args.train_mode}_NoiseStd-{args.noise_std}_roc.json",
        )
        result = {
            "fpr": list(fpr.astype(float)),
            "tpr": list(tpr.astype(float)),
            "threshoods": list(thresholds.astype(float)),
        }
        with open(log_file, "w") as f:
            f.write(json.dumps(result, indent=4))

    def save_emb(self):
        """Compute and save the compressed embeddings."""
        args = self.args

        mod_miss_masks = np.concatenate(self.gt_mod_miss_mask_list, axis=0)
        compressed_emb = self.miss_detector.compute_tsne_emb()
        assert len(mod_miss_masks) == len(compressed_emb)

        result = {
            "miss_masks": mod_miss_masks.tolist()[:500],
            "embs": compressed_emb.tolist()[:500],
        }
        log_file = os.path.join(
            args.log_path,
            f"{args.miss_detector}_{args.train_mode}_NoiseStd-{args.noise_std}_embs.json",
        )
        with open(log_file, "w") as f:
            f.write(json.dumps(result, indent=4))

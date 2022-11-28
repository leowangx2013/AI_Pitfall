import os
import time
import math
import torch
import torch.nn as nn

from torch.nn import TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, out_channel, dropout=0.1, max_len=20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, out_channel, 2) * (-math.log(10000.0) / out_channel))
        pe = torch.zeros(max_len, 1, out_channel)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[: x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)


class FusionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, attention_dropout_rate):
        """Normalization + Attention + Dropout

        Args:
            embed_dim (_type_): _description_
            num_heads (_type_): _description_
            dropout_rate (_type_): _description_
            attention_dropout_rate (_type_): _description_
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout_rate, batch_first=True)

    def forward(self, inputs):
        """Sensor fusion by attention

        Args:
            inputs (_type_): [batch_size, time_interval, mod_num/loc_num, feature_channels]

        Returns:
            _type_: [batch_size, time_interval, feature_channels]
        """
        # [b, i, s, c] -- > [b * i, s, c]
        b, i, s, c = inputs.shape
        inputs = torch.reshape(inputs, (b * i, s, c))

        # norm
        x = self.norm1(inputs)

        # attention-based fusion, [b * i, 1, c]
        mean_query = torch.mean(x, dim=1, keepdim=True)
        x, attn_weights = self.mha(mean_query, x, x)

        # [b * i, 1, c] --> [b, i, c]
        y = torch.reshape(x, (b, i, c))

        return y


class Transformer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["Transformer"]
        self.device = args.device
        self.loc_modalities = args.dataset_config["loc_modalities"]
        self.locations = args.dataset_config["location_names"]
        self.loc_mod_spectrum_len = args.dataset_config["loc_mod_spectrum_len"]
        self.loc_mod_in_channels = args.dataset_config["loc_mod_in_channels"]
        self.num_segments = args.dataset_config["num_segments"]

        # Single mod,  [b, i, s*c]
        self.loc_mod_feature_extraction_layers = nn.ModuleDict()
        for loc in self.locations:
            self.loc_mod_feature_extraction_layers[loc] = nn.ModuleDict()
            for mod in self.loc_modalities[loc]:
                spectrum_len = self.loc_mod_spectrum_len[loc][mod]
                feature_channels = self.loc_mod_in_channels[loc][mod]
                module_list = [nn.Linear(spectrum_len * feature_channels, self.config["loc_mod_out_channels"])] + [
                    TransformerEncoderLayer(
                        d_model=self.config["loc_mod_out_channels"],
                        nhead=self.config["loc_mod_head_num"],
                        dim_feedforward=self.config["loc_mod_out_channels"],
                        dropout=self.config["dropout_ratio"],
                        batch_first=True,
                    )
                    for _ in range(self.config["loc_mod_block_num"])
                ]
                self.loc_mod_feature_extraction_layers[loc][mod] = nn.Sequential(*module_list)

        # Single loc, [b, i, c]
        self.mod_fusion_layers = nn.ModuleDict()
        self.loc_feature_extraction_layers = nn.ModuleDict()
        for loc in self.locations:
            self.mod_fusion_layers[loc] = FusionBlock(
                self.config["loc_mod_out_channels"],
                self.config["loc_head_num"],
                self.config["dropout_ratio"],
                self.config["dropout_ratio"],
            )
            module_list = [nn.Linear(self.config["loc_mod_out_channels"], self.config["loc_out_channels"])] + [
                TransformerEncoderLayer(
                    d_model=self.config["loc_out_channels"],
                    nhead=self.config["loc_head_num"],
                    dim_feedforward=self.config["loc_out_channels"],
                    dropout=self.config["dropout_ratio"],
                    batch_first=True,
                )
                for _ in range(self.config["loc_block_num"])
            ]
            self.loc_feature_extraction_layers[loc] = nn.Sequential(*module_list)

        # Single interval, [b, i, c]
        self.loc_fusion_layer = FusionBlock(
            self.config["loc_out_channels"],
            self.config["sample_head_num"],
            self.config["dropout_ratio"],
            self.config["dropout_ratio"],
        )
        module_list = [nn.Linear(self.config["loc_out_channels"], self.config["sample_out_channels"])] + [
            TransformerEncoderLayer(
                d_model=self.config["sample_out_channels"],
                nhead=self.config["sample_head_num"],
                dim_feedforward=self.config["sample_out_channels"],
                dropout=self.config["dropout_ratio"],
                batch_first=True,
            )
            for _ in range(self.config["sample_block_num"])
        ]
        self.sample_feature_extraction_layer = nn.Sequential(*module_list)

        # Time fusion, [b, c]
        self.time_fusion_layer = FusionBlock(
            self.config["sample_out_channels"],
            self.config["sample_head_num"],
            self.config["dropout_ratio"],
            self.config["dropout_ratio"],
        )

        # Classification
        self.class_layer = nn.Sequential(
            nn.Linear(self.config["sample_out_channels"], self.config["fc_dim"]),
            nn.GELU(),
            nn.Linear(self.config["fc_dim"], args.dataset_config["num_classes"]),
            nn.Sigmoid() if args.multi_class else nn.Softmax(dim=1),
        )

    def forward(self, x, miss_simulator):
        # Step 1: Feature-level fusion, [b, i, s, c]
        org_loc_mod_features = dict()
        for loc in self.locations:
            org_loc_mod_features[loc] = []
            for mod in self.loc_modalities[loc]:
                # [b, c, i, spectrum] -- > [b, i, spectrum, c]
                inputs = torch.permute(x[loc][mod], [0, 2, 3, 1]).to(self.device)
                b, i, s, c = inputs.shape
                inputs = torch.reshape(inputs, (b, i, s * c))
                org_loc_mod_features[loc].append(self.loc_mod_feature_extraction_layers[loc][mod](inputs))
            org_loc_mod_features[loc] = torch.stack(org_loc_mod_features[loc], dim=2)

        # Step 2: Miss modality simultator.
        recon_loc_mod_features, dt_loc_miss_masks = miss_simulator(org_loc_mod_features, BISC_order=True)

        # Step 3: Fusion + Classification layers
        recon_fused_loc_features, recon_logits = self.classification_forward(recon_loc_mod_features)

        # Step 4: Compute the handler loss
        handler_loss_feature_level = miss_simulator.miss_handler.loss_feature_level
        if handler_loss_feature_level == "mod":
            handler_loss = miss_simulator.miss_handler.handler_loss_all_locs(
                org_loc_mod_features,
                recon_loc_mod_features,
                dt_loc_miss_masks,
            )
        elif handler_loss_feature_level == "loc":
            org_fused_loc_features, org_logits = self.classification_forward(org_loc_mod_features)
            handler_loss = miss_simulator.miss_handler.handler_loss_all_locs(
                org_fused_loc_features,
                recon_fused_loc_features,
                dt_loc_miss_masks,
            )
        elif handler_loss_feature_level == "sample":
            org_fused_loc_features, org_logits = self.classification_forward(org_loc_mod_features)
            handler_loss = miss_simulator.miss_handler.handler_loss_all_locs(
                org_logits,
                recon_logits,
                dt_loc_miss_masks,
            )
        elif handler_loss_feature_level == "mod+sample":
            org_fused_loc_features, org_logits = self.classification_forward(org_loc_mod_features)
            handler_loss = miss_simulator.miss_handler.handler_loss_all_locs(
                org_loc_mod_features,
                org_logits,
                recon_loc_mod_features,
                recon_logits,
                dt_loc_miss_masks,
            )

        return recon_logits, handler_loss

    def classification_forward(self, loc_mod_features):
        """Separate the fusion and classification layer forward into this function.

        Args:
            loc_mod_features (_type_): dict of {loc: loc_features}
            return_fused_features (_type_, optional): Flag indicator. Defaults to False.
        """
        # Step 3: Modality-level fusion
        loc_fused_features = {}
        for loc in loc_mod_features:
            loc_fused_features[loc] = self.mod_fusion_layers[loc](loc_mod_features[loc])

        # Step 3: Location feature extraction, [b, i, s, c]
        loc_features = []
        for loc in loc_mod_features:
            outputs = self.loc_feature_extraction_layers[loc](loc_fused_features[loc])
            loc_features.append(outputs)
        loc_features = torch.stack(loc_features, dim=2)

        # Step 4: Location-level fusion, [b, i, c]
        interval_features = self.loc_fusion_layer(loc_features)
        interval_features = self.sample_feature_extraction_layer(interval_features)
        interval_features = torch.unsqueeze(interval_features, dim=1)

        # Step 5: Time fusion
        sample_features = self.time_fusion_layer(interval_features)
        sample_features = torch.flatten(sample_features, start_dim=1)

        # Step 5: Classification
        outputs = torch.flatten(sample_features, start_dim=1)
        logits = self.class_layer(outputs)

        return loc_fused_features, logits


if __name__ == "__main__":
    pass

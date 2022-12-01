import torch
import torch.nn as nn

from models.ResidualAutoencoder import RA_Encoder, RA_Decoder


class VAE(nn.Module):
    def __init__(self, args, in_channels, config, sensors=None) -> None:
        """The VAE model for one (loc, mod)"""
        super().__init__()
        self.args = args
        self.config = config
        self.fuse_time_flag = self.config["conv_kernel_size"][0] > 1

        # store the dimensions
        self.interval = 1 if args.model == "ResNet" else self.config["interval_num"]
        self.sensors = sensors if sensors else self.config["sensors"]

        """Define the architecture"""
        self.encoder = RA_Encoder(
            sensors,
            in_channels,
            self.config["conv_inter_channels"],
            self.config["conv_out_channels"],
            self.config["conv_kernel_size"],
            self.config["fc_dim"],
            self.interval,
            self.config["conv_inter_layer_num"],
            self.fuse_time_flag,
        )

        self.fc_mu = nn.Linear(self.config["fc_dim"], self.config["fc_dim"])
        self.fc_var = nn.Linear(self.config["fc_dim"], self.config["fc_dim"])

        self.decoder_fc = nn.Linear(self.config["fc_dim"], self.config["fc_dim"])
        self.decoder = RA_Decoder(
            sensors,
            in_channels,
            self.config["conv_inter_channels"],
            self.config["conv_out_channels"],
            self.config["conv_kernel_size"],
            self.config["fc_dim"],
            self.interval,
            self.config["conv_inter_layer_num"],
            self.fuse_time_flag,
            self.config["output_activation"],
        )

        # loss funcitons
        self.recon_loss_func = nn.MSELoss()
        self.kl_loss_weight = self.config["kl_divergence_weight"]

    def encode(self, x):
        """Encoding an input with shape [b, c, i, s]"""
        z_enc = self.encoder(x)

        """Separate the mu and var components"""
        mu = self.fc_mu(z_enc)
        log_var = self.fc_var(z_enc)

        return mu, log_var

    def decode(self, z):
        """Decode an latent variable of shae [b, c] for each sensor"""
        x_recon = self.decoder_fc(z)
        x_recon = self.decoder(x_recon)

        return x_recon

    def reparameterize(self, mu, log_var):
        """Reparameterization trick in VAE."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Generate the loc-mod missing flags"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_out = self.decode(z)

        return x_out, z, mu, log_var

    def vae_loss(self, x_in, x_recon, mu, log_var):
        """Compute the VAE loss."""
        recon_loss = self.recon_loss_func(x_recon, x_in)
        kl_divergence_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        loss = recon_loss + self.kl_loss_weight * kl_divergence_loss

        return loss

import torch
import torch.nn as nn

from models.ConvModules import ConvLayer2D

"""
Note:
    1) This autoencoder is used in many detectors and handlers, so we create a separate file for better reference.
"""


class RA_Encoder(nn.Module):
    def __init__(
        self,
        sensor,
        in_channels,
        inter_channels,
        out_channels,
        kernel_size,
        fc_dim,
        interval,
        conv_inter_layer_num,
        fuse_time_flag=False,
    ) -> None:
        """The encoder of the residual autoencoder (RA).
        Structure: conv_in + conv_inter + conv_out + fc
        Conv layers are used to reduce the dimension with the same output channel, but stride 2.
        Returns:
            _type_: _description_
        """
        super().__init__()
        # flag for fuse time dimension
        self.fuse_time_flag = fuse_time_flag

        """Define the architecture"""
        self.conv_layer_in = ConvLayer2D(
            in_channels,
            inter_channels,
            kernel_size,
            stride=1,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dropout_ratio=0,
        )

        self.conv_layers_inter = nn.ModuleList()
        for i in range(conv_inter_layer_num):
            conv_layer = ConvLayer2D(
                inter_channels,
                inter_channels,
                kernel_size,
                stride=1,
                padding="same",
                padding_mode="zeros",
                bias=True,
                dropout_ratio=0,
            )
            self.conv_layers_inter.append(conv_layer)

        self.conv_layer_out = ConvLayer2D(
            inter_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dropout_ratio=0,
        )

        # in_features, out_features
        if self.fuse_time_flag:
            self.fc = nn.Linear(sensor * interval * out_channels, fc_dim)
        else:
            self.fc = nn.Linear(sensor * out_channels, fc_dim)

    def forward(self, x):
        """The forward function of the RA_Encoder.

        Args:
            x (_type_): [b, c_in, i, sensor]

        Returns:
            _type_: [b * i, fc_dim]
        """
        # conv layers
        conv_out = self.conv_layer_in(x)

        # inter conv layers
        for conv_layer in self.conv_layers_inter:
            conv_out = conv_layer(conv_out) + conv_out

        # out conv layer
        conv_out = self.conv_layer_out(conv_out)

        # (b, c, i, s) --> (b, i, c, s) --> (b * i, c * s)
        conv_out = conv_out.permute(0, 2, 1, 3)
        b, i, c, s = conv_out.shape
        if self.fuse_time_flag:
            conv_out = torch.reshape(conv_out, (b, i * c * s))
        else:
            conv_out = torch.reshape(conv_out, (b * i, c * s))

        # fc layer, (b * i, fc_dim) or (b, fc_dim)
        fc_out = self.fc(conv_out)

        return fc_out


class RA_Decoder(nn.Module):
    def __init__(
        self,
        sensor,
        in_channels,
        inter_channels,
        out_channels,
        kernel_size,
        fc_dim,
        interval,
        conv_inter_layer_num,
        fuse_time_flag=False,
        output_activation="GELU",
    ) -> None:
        """The decoder of the residual autoencoder (RA).
        Structure:  fc + deconv1 + deconv2
        Returns:
            _type_: _description_
        """
        super().__init__()
        self.fuse_time_flag = fuse_time_flag

        # store the dimensions
        self.c = out_channels
        self.i = interval
        self.s = sensor

        # in_features, out_features
        if self.fuse_time_flag:
            self.fc = nn.Linear(fc_dim, sensor * out_channels * interval)
        else:
            self.fc = nn.Linear(fc_dim, sensor * out_channels)

        self.deconv_layer_in = ConvLayer2D(
            out_channels,
            inter_channels,
            kernel_size,
            stride=1,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dropout_ratio=0,
        )

        self.deconv_layers_inter = nn.ModuleList()
        for i in range(conv_inter_layer_num):
            deconv_layer = ConvLayer2D(
                inter_channels,
                inter_channels,
                kernel_size,
                stride=1,
                padding="same",
                padding_mode="zeros",
                bias=True,
                dropout_ratio=0,
            )
            self.deconv_layers_inter.append(deconv_layer)

        self.deconv_layer_out = ConvLayer2D(
            inter_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding="same",
            padding_mode="zeros",
            bias=True,
            dropout_ratio=0,
            activation=output_activation,
        )

    def forward(self, x):
        """The forward function of the RA_Decoder.

        Args:
            x (_type_): (b * i, fc_dim)

        Returns:
            _type_: Same as the input dimension, [b, c, intervals, sensor]
        """
        # fc layer first
        fc_out = self.fc(x)

        # reshape: (b * i, c * s) --> (b, i, c, s) --> (b, c, i, s)
        fc_out = torch.reshape(fc_out, (-1, self.i, self.c, self.s))
        fc_out = fc_out.permute(0, 2, 1, 3)

        # deconv layer in
        deconv_out = self.deconv_layer_in(fc_out)

        # deconv layer inter
        for deconv_layer in self.deconv_layers_inter:
            deconv_out = deconv_layer(deconv_out) + deconv_out

        # deconv layer out
        deconv_out = self.deconv_layer_out(deconv_out)

        return deconv_out


class ResidualAutoencoder(nn.Module):
    def __init__(self, args, in_channels, config, sensors=None) -> None:
        super().__init__()

        self.args = args
        self.config = config

        # fuse time flag
        self.fuse_time_flag = self.config["conv_kernel_size"][0] > 1

        # store the dimensions
        self.interval = 1 if args.model == "ResNet" else self.config["interval_num"]
        self.sensors = sensors if sensors else self.config["sensors"]

        self.encoder = RA_Encoder(
            self.sensors,
            in_channels,
            config["conv_inter_channels"],
            config["conv_out_channels"],
            config["conv_kernel_size"],
            config["fc_dim"],
            self.interval,
            config["conv_inter_layer_num"],
            self.fuse_time_flag,
        )

        self.decoder = RA_Decoder(
            self.sensors,
            in_channels,
            config["conv_inter_channels"],
            config["conv_out_channels"],
            config["conv_kernel_size"],
            config["fc_dim"],
            self.interval,
            config["conv_inter_layer_num"],
            self.fuse_time_flag,
        )

    def forward(self, x, return_emb=False):
        """Forward function"""
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        out_x = x + decoded_x

        if return_emb:
            return out_x, encoded_x
        else:
            return out_x

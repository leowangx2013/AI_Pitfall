import torch
import torch.nn as nn


class MeanFusionBlock(nn.Module):
    def __init__(self) -> None:
        """The initialization of the mean fusion block.
        Stureture:
        """
        super().__init__()

    def forward(self, x):
        """The forward function of the MeanFusionBlock.

        Args:
            x (_type_): list of [b, c, intervals, sensors]
        Return:
            [b, 1, i, c]
        """
        # mean out
        mean_out = torch.mean(x, dim=3, keepdim=False)

        # flatten and move c to spectral samples, [b, c, i] --> [b, 1, i, s]
        mean_out = mean_out.permute(0, 2, 1)
        mean_out = torch.unsqueeze(mean_out, dim=1)

        return mean_out


class SelfAttentionFusionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, bias=True) -> None:
        """The initialization of the self-attention fusion block.
        Structure:
        """
        super().__init__()

        # define the self attention head
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=True)

    def forward(self, x):
        """The forward function of the SelfAttentionBlock.

        Args:
            x (_type_): list of [b, c, intervals, sensors]
        Output:
            [b, c, i] or [b, 1, i, c]
        """
        # concat and exchange dimension: [b, c, i, s] --> [b, i, s, c] --> [b * i, s, c]
        x = x.permute(0, 2, 3, 1)
        b, i, s, c = x.shape
        x = torch.reshape(x, (b * i, s, c))

        # Step 1: Calculate the mean query, shape: [b * i, 1 (one query only), c]
        mean_query = torch.mean(x, dim=1, keepdim=True)

        # Step 2: Attention, attention out: [b, c, i]
        attn_out, attn_weights = self.attention(mean_query, x, x, need_weights=True)
        attn_out = attn_out.reshape(attn_out, (b, i, 1, c))
        attn_out = torch.squeeze(attn_out, dim=2)
        attn_out = torch.reshape(attn_out, (b, i, c))
        attn_out = attn_out.permute(0, 2, 1)

        # flatten and move c to spectral samples, [b, c, i] --> [b, 1, i, s (=c)]
        attn_out = attn_out.permute(0, 2, 1)
        attn_out = torch.unsqueeze(attn_out, dim=1)

        return attn_out

"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from taming.modules.transformer.decoder import Decoder
from taming.clip.model import quantTransformer as Encoder


class maskTransformer(nn.Module):

    def __init__(self, grid: int, width: int,  n_layers: int, n_head: int, d_model: int, max_len: int,
                 ffn_hidden: int, drop_prob: float, device: str):
        super().__init__()


        self.device = device
        self.encoder = Encoder(
                grid=grid,
                width=width,
                layers=n_layers,
                heads=n_head,
                output_dim=d_model
            )

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, img, text, mask):
        enc_src = self.encoder(img, return_all=True)
        output = self.decoder(text, enc_src, mask)
        return output, enc_src


if __name__ == '__main__':

    a = torch.randn((2, 3, 3))
    c = torch.randn((2, 3, 3))
    b = torch.tensor([[[1.],
         [1.],
                       [0.]]])
    # print(b, a, c)
    # print((a[0, 0] - c[0, 0]) **2 / 2.0)

    mse_loss = (a*b - c*b) ** 2  # B * N * d
    mse_loss = mse_loss.mean(-1).sum(-1) / 2.0
    print(mse_loss.mean())
    # print(torch.norm(a, p=2, dim=-1))
    # a = a / torch.norm(a, p=2, dim=-1).unsqueeze(-1)
    # c = c / torch.norm(c, p=2, dim=-1).unsqueeze(-1)
    #
    # print(a.matmul(c.T))
    # print(torch.nn.functional.cosine_similarity(a, c))
    print(torch.nn.functional.mse_loss(a[:, :2], c[:, :2],))








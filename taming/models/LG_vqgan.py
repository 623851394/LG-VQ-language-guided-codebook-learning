import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.checkpoint import checkpoint
# from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer

from taming.clip import clip
from taming.clip.model import quantTransformer
from taming.modules.transformer.maskTransformer import maskTransformer

import argparse, os, sys, datetime, glob, importlib


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class MlmLayer(nn.Module):

    def __init__(self, feat_emb_dim, word_emb_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(word_emb_dim)
        self.bias = nn.Parameter(torch.zeros(1, 1, vocab_size))

    def forward(self, x, word_embeddings):
        mlm_hidden = self.fc(x)
        mlm_hidden = self.gelu(mlm_hidden)
        mlm_hidden = self.ln(mlm_hidden)
        word_embeddings = word_embeddings.transpose(0, 1)
        logits = torch.matmul(mlm_hidden, word_embeddings)
        logits = logits + self.bias
        return logits

class wrsLayer(nn.Module):

    def __init__(self, quant_dim, text_dim):
        super().__init__()
        self.fc = nn.Linear(quant_dim, text_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(text_dim)
        # self.bias = nn.Parameter(torch.zeros(1, 1, text_dim))

        torch.nn.init.normal_(self.fc.weight, std=0.01)


    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        output = self.ln(x)
        # logits = logits + self.bias
        return output



class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.type(torch.int64).detach()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        # print(nll_loss)
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.dim = embed_dim
        self.text_dim = 512
        self.clip_text_truncate_len = ddconfig['clip_text_truncate_len']

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        # Mask Prediction loss
        # Mask Transformer
        self.mskTransformer = maskTransformer(
            grid=16,
            width=self.dim,
            n_layers=ddconfig['mask_transformer_layer'],
            n_head=ddconfig['mask_transformer_head'],
            d_model=self.text_dim,
            max_len=ddconfig['clip_text_truncate_len'],
            ffn_hidden=self.text_dim,
            drop_prob=ddconfig['drop_prob'],
            device=self.device,
        )
        #
        self.mask_learned_parameter = torch.nn.Parameter(
            torch.randn((1, ddconfig['clip_text_truncate_len'], self.text_dim)))
        torch.nn.init.normal_(self.mask_learned_parameter, std=0.01)
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        self.mseloss = nn.MSELoss()
        self.wrsr_fc = wrsLayer(self.dim, self.text_dim)


        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        print("loading clip model....")
        self.clip_model, _ = clip.load("ViT-B-32.pt", device=self.device)
        for name, param in self.clip_model.named_parameters():
            param.requires_grad = False

        self.mlm_layer = MlmLayer(feat_emb_dim=self.text_dim, word_emb_dim=self.text_dim,
                                  vocab_size=self.clip_model.vocab_size)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        # print("encoder infor:", type(x), x.shape)
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def global_infor_sup(self, quant, last_text_feature):

        quant_to_text_feature = quant   #

        # norm
        quant_to_text_feature = quant_to_text_feature / torch.norm(quant_to_text_feature, dim=-1).unsqueeze(-1)
        last_text_feature = last_text_feature / torch.norm(last_text_feature, dim=-1).unsqueeze(-1)
        last_text_feature = last_text_feature.detach()
        # # cosine_similarity
        pos_swrse = (quant_to_text_feature * last_text_feature).sum(-1)


        neg_swrse = torch.exp(torch.matmul(quant_to_text_feature, last_text_feature.T)).sum(-1)
        # NCEloss
        nce_loss = -torch.log(torch.exp(pos_swrse) / (neg_swrse + 0.000001))

        return nce_loss

    def maskPrediction(self, quant, all_text_features, text_mask, mask_padding, ge_indices):

        mask_text_features = torch.where(text_mask.unsqueeze(-1).repeat(1, 1, self.text_dim) == 0,
                                         self.mask_learned_parameter, all_text_features)
        #
        mask_padding_text_features = mask_text_features * mask_padding.unsqueeze(-1)
        # B * 1 * N
        mask_padding_ = mask_padding.unsqueeze(1)
        # B * N * N
        mask_padding_ = torch.matmul(mask_padding_, mask_padding_.permute(0, 2, 1))  #
        # B * N * d
        output, vision_token = self.mskTransformer(quant, mask_padding_text_features, mask_padding_)
        # B * N
        mask_pre = (~text_mask.bool()) * (mask_padding.bool())  #
        mask_pre = mask_pre.float()

        # CE loss
        mskPre_loss = self.crossEntropy(ge_indices, output, mask_pre)

        vision_global_token = vision_token[:, 0]

        return mskPre_loss, vision_global_token, vision_token[:, 1:]

    def crossEntropy(self, gt_indices, x, mask):

        word_embeddings = self.clip_model.token_embedding.weight.data.detach()
        logits = self.mlm_layer(x, word_embeddings)  # B * N * K

        bsz, seq_len = gt_indices.size()

        loss = self.criterion(logits.reshape(bsz * seq_len, -1),
                              gt_indices.reshape(bsz * seq_len))
        loss = loss.reshape(bsz, seq_len)
        loss = (loss * mask).sum() / (mask.sum() + 0.000005)  # mean loss on removed patches
        return loss

    def text_supervise(self, quant, all_text_features, last_text_feature, text_mask, mask_padding, text_tokens):

        mskPre_loss, vision_global_token, vision_token = self.maskPrediction(quant, all_text_features, text_mask, mask_padding, text_tokens)
        mskPre_loss = mskPre_loss.mean()
        nce_loss = self.global_infor_sup(vision_global_token, last_text_feature).mean()


        wrsr_loss = self.wrsrelation_sup(vision_token, quant, all_text_features, mask_padding)

        return nce_loss, mskPre_loss, wrsr_loss

    def wrsrelation_sup(self, vision_token, quant, all_text_features, mask_padding):

        x = quant.reshape(quant.shape[0], quant.shape[1], -1)  # B * d * 16 * 16
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        quant_ = self.wrsr_fc(x)

        norm_ = torch.norm(all_text_features, p=2, dim=-1)
        all_text_features = all_text_features / norm_.unsqueeze(-1)

        vision_token_norm = vision_token / torch.norm(vision_token, p=2, dim=-1).unsqueeze(-1)
        token_text_sim = torch.matmul(all_text_features, vision_token_norm.permute(0, 2, 1))  # B * seq * 256
        value, indices = token_text_sim.max(-1)


        text_to_quant = torch.zeros(vision_token_norm.shape[0], all_text_features.shape[1], self.text_dim).to(self.device)
        for i in range(all_text_features.shape[1]):
            text_to_quant[torch.arange(vision_token_norm.shape[0]).view(-1, 1), i, :] = quant_[
                torch.arange(vision_token_norm.shape[0]).view(-1, 1), indices[:, i].reshape(-1, 1)]


        # text_relation
        all_text_features = all_text_features * mask_padding.unsqueeze(-1)
        q = torch.matmul(all_text_features, all_text_features.permute(0, 2, 1))  # B * seq * seq

        # code_relation
        text_to_quant = text_to_quant * value.unsqueeze(-1)  # B * seq * grid^2
        norm_ = torch.norm(text_to_quant, p=2, dim=-1).unsqueeze(-1)
        text_to_quant = text_to_quant / norm_
        text_to_quant = text_to_quant * mask_padding.unsqueeze(-1)
        p = torch.matmul(text_to_quant, text_to_quant.permute(0, 2, 1))

        relation_loss = self.mseloss(p, q.detach())

        return relation_loss

    def forward(self, input_img):
        quant, diff, _ = self.encode(input_img)

        dec = self.decode(quant)

        return dec, diff, quant

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def get_text_input(self, batch):
        x = batch["text"]
        # clip token
        text_tokens, text_mask = clip.tokenize(x, context_length=self.clip_text_truncate_len)
        text_tokens, text_mask = text_tokens.to(self.device), text_mask.to(self.device)
        mask_padding = (text_tokens != 0).float()
        with torch.no_grad():
            all_text_features, last_text_feature = self.clip_model.encode_text(text_tokens)

        return all_text_features, last_text_feature, text_mask, mask_padding, text_tokens

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        # print("x shape :", x.shape)
        all_text_features, last_text_feature, text_mask, mask_padding, text_tokens = self.get_text_input(batch)

        xrec, qloss, quant = self(x)

        nceloss, mskPre_loss, wrs_loss = self.text_supervise(quant, all_text_features, last_text_feature, text_mask,
                                                   mask_padding, text_tokens)
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            nceloss=nceloss, mskloss=mskPre_loss, wrsloss=wrs_loss,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                nceloss=nceloss, mskloss=mskPre_loss, wrsloss=wrs_loss,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, _ = self(x)
        xrec, qloss, quant = self(x)


        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, isValid=True,
                                        last_layer=self.get_last_layer(), split="val")

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["val/rec_loss"]
        self.log_dict(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate

        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()) +
                                  list(self.mskTransformer.parameters()) +
                                  [self.mask_learned_parameter] +
                                  list(self.wrsr_fc.parameters()) +
                                  list(self.mlm_layer.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()) +
                                  list(self.mskTransformer.parameters()) +
                                  [self.mask_learned_parameter] +
                                  list(self.wrsr_fc.parameters()) +
                                  list(self.mlm_layer.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        all_text_features, last_text_feature, text_mask, mask_padding, text_tokens = self.get_text_input(batch)

        xrec, qloss, quant = self(x)

        nceloss, mskPre_loss, wrs_loss = self.text_supervise(quant, all_text_features, last_text_feature, text_mask,
                                                             mask_padding, text_tokens)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train", nceloss=nceloss, mskloss=mskPre_loss, wrsloss=wrs_loss)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss,_ = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val", isValid=True)
        total_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["val/rec_loss"]
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                     list(self.decoder.parameters()) +
                                     list(self.quantize.parameters()) +
                                     list(self.quant_conv.parameters()) +
                                     list(self.post_quant_conv.parameters()),
                                     lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)  # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)

        all_text_features, last_text_feature, text_mask, mask_padding, text_tokens = self.get_text_input(batch)

        xrec, qloss, quant = self(x)

        nceloss, mskPre_loss, wrs_loss = self.text_supervise(quant, all_text_features, last_text_feature, text_mask,
                                                   mask_padding, text_tokens)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            nceloss=nceloss, mskloss=mskPre_loss, wrsloss=wrs_loss,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                nceloss=nceloss,
                                                mskloss=mskPre_loss,
                                                last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        # xrec, qloss = self(x, return_pred_indices=True)
        xrec, qloss, _ = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, isValid=True,
                                        last_layer=self.get_last_layer(), split="val")

        # discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
        #                                     last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        del log_dict_ae["val/rec_loss"]
        # self.log("val/aeloss", aeloss,
        #          prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        # self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class EMAVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )
        self.quantize = EMAVectorQuantizer(n_embed=n_embed,
                                           embedding_dim=embed_dim,
                                           beta=0.25,
                                           remap=remap)

    def configure_optimizers(self):
        lr = self.learning_rate
        # Remove self.quantize from parameter list since it is updated via EMA
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []





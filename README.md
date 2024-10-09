# LG-VQ: Language Guided Codebook Learning

This repository contains the code for the paper:

[LG-VQ: Language-Guided Codebook Learning](https://arxiv.org/pdf/2405.14206)

Guotao Liang, Baoquan Zhang, Yaowei Wang, Xutao Li, Yunming Ye, Huaibin Wang, Chuyao Luo, Kola Ye, Linfeng Luo.

Accepted by NeurIPS 2024

## Abstract
Vector quantization (VQ) is a key technique in high-resolution and high-fidelity image synthesis, which aims to learn a codebook to encode an image with a sequence of discrete codes and then generate an image in an auto-regression manner. Although existing methods have shown superior performance, most methods prefer to learn a single-modal codebook (\emph{e.g.}, image), resulting in suboptimal performance when the codebook is applied to multi-modal downstream tasks (\emph{e.g.}, text-to-image, image captioning) due to the existence of modal gaps. In this paper, we propose a novel language-guided codebook learning framework, called LG-VQ, which aims to learn a codebook that can be aligned with the text to improve the performance of multi-modal downstream tasks. Specifically, we first introduce pre-trained text semantics as prior knowledge, then design two novel alignment modules (\emph{i.e.}, Semantic Alignment Module, and Relationship Alignment Module) to transfer such prior knowledge into codes for achieving codebook text alignment. In particular, our LG-VQ method is model-agnostic, which can be easily integrated into existing VQ models. Experimental results show that our method achieves superior performance on reconstruction and various multi-modal downstream tasks. 

## Citation

If you use this code for your research, please cite our paper:
```bibtex
@article{liang2024lg,
  title={LG-VQ: Language-Guided Codebook Learning},
  author={Liang, Guotao and Zhang, Baoquan and Wang, Yaowei and Li, Xutao and Ye, Yunming and Wang, Huaibin and Luo, Chuyao and Ye, Kola and others},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```

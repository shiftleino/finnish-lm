# Language models trained for Finnish
This repository contains implementations for Finnish language models using various state-of-the-art techniques. Just for fun :)

Heavily inspired by and partly following the great Andrej Karpathy [nanoGPT-series](https://github.com/karpathy/nanoGPT) (see e.g., [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)).

## 1. Char-level LM trained with Kalevala
As the first language model we trained a character-level language model with the Finnish national epic Kalevala and followed how the different techniques contributed to the model performance. The version specifications are in the format of (number of layers, model dimension, number of attention heads, block size). In training, we use early stopping with patience of 10. All the runs use a block size of 512 tokens for evaluating the validation loss. For models with under 30M parameters we use a Macbook and Metal Performance Shaders to train the model, while for models with over 30M parameters we use an AMD GPU and ROCm.

 | Model version | Non-linearity | Positional information | Training loss | Validation loss | Parameters | Generations |
 |:----------:|:----------:|:----------:|:----------:|:----------:| :----------:| :----------:|
 | Pre-norm Attention-only (1, 1024, 16, 512) | None | None | 1.90191 | 1.91712 | 4M | [Link](./kaleGPTs/generations/kalegpt-attention-no-pos-1-1024-16-512.txt) |
 | Pre-norm Transformer (1, 1024, 16, 512) | ReLU | None | 1.90350 | 1.93926 | 12M | [Link](./kaleGPTs/generations/kalegpt-transformer-no-pos-1-1024-16-512.txt) |
 | Pre-norm Transformer (1, 1024, 16, 512) | ReLU | Embeddings | 0.69168 | 1.17659 | 13M | [Link](./kaleGPTs/generations/kalegpt-transformer-relu-1-1024-16-512.txt) |
| Pre-norm Transformer (1, 1024, 16, 512) | Leaky ReLU | Embeddings | 0.75599 | 1.16501 | 13M | [Link](./kaleGPTs/generations/kalegpt-transformer-leaky-relu-1-1024-16-512.txt) |
| Pre-norm Transformer (1, 1024, 16, 512) | GELU | Embeddings | 0.71472 | 1.18524 | 13M | [Link](./kaleGPTs/generations/kalegpt-transformer-gelu-1-1024-16-512.txt) |
| Pre-norm Transformer (3, 768, 12, 512) | ReLU | Embeddings | 0.49662 | 1.15837 | 22M | [Link](./kaleGPTs/generations/kalegpt-relu-3-768-12-512.txt) |
| Pre-norm Transformer (3, 768, 12, 512) | GELU | Embeddings | 0.43291 | 1.23698 | 22M | [Link](./kaleGPTs/generations/kalegpt-gelu-3-768-12-512.txt) |
| Pre-norm Transformer with Dropout (3, 768, 12, 512) | GELU | Embeddings | 0.82735 | 1.01666 | 22M | [Link](./kaleGPTs/generations/kalegpt-dropout-gelu-3-768-12-512.txt) |
| Pre-norm Transformer with Dropout (3, 768, 12, 512) | ReLU | Embeddings | 0.81985 | 1.01711 | 22M | [Link](./kaleGPTs/generations/kalegpt-dropout-relu-3-768-12-512.txt) |
| Pre-RMSNorm Transformer with Dropout (3, 768, 12, 512) | GELU | Embeddings | 0.82009 | 1.02458 | 22M | [Link](./kaleGPTs/generations/kalegpt-dropout-rmsnorm-gelu-3-768-12-512.txt) |
| Pre-RMSNorm Transformer with Dropout (3, 768, 12, 512) | SwiGLU | Embeddings | 0.83574 | 1.07487 | 29M | [Link](./kaleGPTs/generations/kalegpt-dropout-rmsnorm-swiglu-3-768-12-512.txt) |
| Pre-RMSNorm Transformer with Dropout (3, 768, 12, 512) | ReLU | ALiBi | 1.00742 | 1.14893 | 21M | [Link](./kaleGPTs/generations/kalegpt-dropout-alibi-rmsnorm-relu-3-768-12-512.txt) |
| Pre-RMSNorm Transformer with Dropout (3, 768, 12, 256) | ReLU | ALiBi | 1.11632 | 1.18132 | 21M | [Link](./kaleGPTs/generations/kalegpt-dropout-alibi-rmsnorm-relu-3-768-12-256.txt) |
| Pre-RMSNorm Transformer with Dropout (3, 768, 12, 512) | ReLU | RoPE | 0.75139 | 1.03126 | 21M | [Link](./kaleGPTs/generations/kalegpt-dropout-rope-rmsnorm-relu-3-768-12-512.txt) |
| Pre-RMSNorm Transformer with Dropout (5, 768, 12, 512) | ReLU | RoPE | 0.73791 | 0.99258 | 36M | [Link](./kaleGPTs/generations/kalegpt-dropout-rope-rmsnorm-relu-5-768-12-512.txt) |
| Pre-RMSNorm Transformer with Dropout (8, 768, 12, 512) | ReLU | RoPE | 0.78057 | 0.97327 | 57M | [Link](./kaleGPTs/generations/kalegpt-dropout-rope-rmsnorm-relu-8-768-12-512.txt) |
| Pre-RMSNorm Transformer with Dropout (12, 768, 12, 512) | ReLU | RoPE | 0.81678 | 0.97046 | 85M | [Link](./kaleGPTs/generations/kalegpt-dropout-rope-rmsnorm-relu-12-768-12-512.txt) |
| Pre-RMSNorm Transformer with Dropout (12, 768, 12, 512) | SwiGLU | RoPE |  | | 85M | [Link](./kaleGPTs/generations/kalegpt-dropout-rope-rmsnorm-swiglu-12-768-12-512.txt) |

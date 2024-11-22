# Language models trained for Finnish
This repository contains implementations for Finnish language models using various state-of-the-art techniques. Just for fun :)

Heavily inspired by and partly following the great Andrej Karpathy [nanoGPT-series](https://github.com/karpathy/nanoGPT) (see e.g., [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)).

## 1. Char-level LM trained with Kalevala
As the first language model we trained a character-level language model with the Finnish national epic Kalevala and followed how the different techniques contributed to the model performance. The version specifications are in the format of (number of layers, model dimension, number of attention heads, block size). In training, we use early stopping with patience of 10.

 | Model version | Non-linearity | Positional information | Training loss | Validation loss | Parameters | Generations |
 |:----------:|:----------:|:----------:|:----------:|:----------:| :----------:| :----------:|
 | Pre-norm Attention-only (1, 1024, 16, 512) | None | None | 1.90191 | 1.91712 | 4M | [Link](./kaleGPT/generations/kalegpt-attention-no-pos-1-1024-16-512.txt) |
 | Pre-norm Transformer (1, 1024, 16, 512) | ReLU | None | 1.90350 | 1.93926 | 12M | [Link](./kaleGPT/generations/kalegpt-transformer-no-pos-1-1024-16-512.txt) |
 | Pre-norm Transformer (1, 1024, 16, 512) | ReLU | Embeddings | 0.69168 | 1.17659 | 13M | [Link](./kaleGPT/generations/kalegpt-transformer-relu-1-1024-16-512.txt) |
| Pre-norm Transformer (1, 1024, 16, 512) | Leaky ReLU | Embeddings | 0.75599 | 1.16501 | 13M | [Link](./kaleGPT/generations/kalegpt-transformer-leaky-relu-1-1024-16-512.txt) |

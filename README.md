# Language models trained for Finnish
This repository contains implementations for Finnish language models using various state-of-the-art techniques. Just for fun :)

Heavily inspired by and partly following the great Andrej Karpathy [nanoGPT-series](https://github.com/karpathy/nanoGPT) (see e.g., [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)).

## 1. Char-level LM trained with Kalevala
As the first language model we trained a character-level language model with the Finnish national epic Kalevala and followed how the different techniques contributed to the model performance. The version specifications are in the format of (number of layers, model dimension, number of attention heads, block size). We use early stopping with patience of 10.

 | Model version | Training loss | Validation loss | Generations
 |:----------:|:----------:|:----------:| :----------:|
 | Attention-only w/o positional information (1, 1024, 16, 512) | 1.97576 | 1.97724 | [Link](./kaleGPT/generations/attention-1-1024-16-512.txt) |
 | Transformer with ReLU w/o positional information (1, 1024, 16, 512) |1.98016 | 1.98373 | [Link](./kaleGPT/generations/transformer-1-1024-16-512.txt) |
 | Transformer with ReLU and positional embedding table (1, 1024, 16, 512) | 0.66615 | 1.26165 | [Link](./kaleGPT/generations/transformer-1-1024-16-512.txt) |

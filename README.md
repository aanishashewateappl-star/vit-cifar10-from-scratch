# vit-cifar10-from-scratch

A Vision Transformer (ViT) implemented from scratch in PyTorch and trained on CIFAR-10. Built as a study project after implementing "Attention Is All You Need" (see [attention-is-all-you-need-study](https://github.com/aanishashewateappl-star/attention-is-all-you-need-study)), following the ideas in *An Image is Worth 16x16 Words* (Dosovitskiy et al.).

## Result

| Dataset | Model | Parameters | Best test accuracy |
|---|---|---|---|
| CIFAR-10 | ViT (patch 4, depth 6, d_model 256) | 4.77M | **82.91%** |

Trained from scratch with no pretraining. "Best test accuracy" is the highest test accuracy over all epochs, so it is slightly optimistic: the checkpoint was selected on the test set rather than a separate validation set.

## Architecture

The image goes through four stages:

```
image (B, 3, 32, 32)
  -> patch embedding + class token + positional embedding   (B, 65, 256)
  -> 6 x encoder block (pre-norm attention + MLP)           (B, 65, 256)
  -> final LayerNorm
  -> take the class token, linear head                      (B, 10)
```
## Settings

| Setting | Value |
|---|---|
| Image size / patch size | 32 / 4 (64 patches + 1 class token) |
| d_model | 256 |
| Heads | 8 |
| MLP hidden size | 1024 |
| Depth | 6 |
| Dropout | 0.1 |
| Epochs | 50 |
| Batch size | 128 |
| Optimizer | AdamW, lr 1e-3, weight decay 0.05 |
| Schedule | 5 epochs linear warmup, then cosine decay |
| Loss | Cross-entropy, label smoothing 0.1 |
| Gradient clipping | 1.0 |
| Augmentation | Random crop (padding 4), random horizontal flip |

## What I learned

- A ViT encoder is the same machinery as the original Transformer encoder, with a new front end (patches) and back end (class token).
- A Vision Transformer has no built-in notion of locality like a CNN, so on a small dataset augmentation matters a lot.

## Reference

Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, 2020.# vit-cifar10-from-scratch

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

1. **Patch embedding:** a `Conv2d` with kernel size = stride = patch size (4) cuts the image into an 8x8 grid of patches and projects each one to a 256-dimensional vector. This is equivalent to flattening each patch and applying a shared linear layer.
2. **Class token and positional embeddings:** a learnable class token is prepended (65 tokens total), and a learnable positional embedding is added so the model knows where each patch came from.
3. **Encoder block:** LayerNorm, multi-head self-attention, residual connection, then LayerNorm, MLP (with GELU), residual connection. There is no mask, since every patch can attend to every other patch. The attention code is reused from my Annotated Transformer implementation.
4. **Classification head:** only the class token's output is used, passed through a single linear layer to give 10 class scores.

## Settings

| Setting | Value |
|---|---|
| Image size / patch size | 32 / 4 (64 patches + 1 class token) |
| d_model | 256 |
| Heads | 8 |
| MLP hidden size | 1024 |
| Depth | 6 |
| Dropout | 0.1 |
| Epochs ⚠️ | 50 |
| Batch size ⚠️ | 128 |
| Optimizer | AdamW, lr 1e-3, weight decay 0.05 |
| Schedule | 5 epochs linear warmup, then cosine decay |
| Loss | Cross-entropy, label smoothing 0.1 |
| Gradient clipping | 1.0 |
| Augmentation | Random crop (padding 4), random horizontal flip |

## Run it

1. Open the notebook in Google Colab and select a GPU runtime (T4 is enough).
2. Run all cells from top to bottom. CIFAR-10 downloads automatically.

The dataset (`data/`) and checkpoints (`*.pt`) are not stored in this repository.

## What I learned

- A ViT encoder is the same machinery as the original Transformer encoder, with a new front end (patches) and back end (class token).
- A Vision Transformer has no built-in notion of locality like a CNN, so on a small dataset augmentation matters a lot.
- Most bugs were shape bugs. Testing each component on random data before assembling the model caught them early.

## Reference

Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, 2020.# vit-cifar10-from-scratch

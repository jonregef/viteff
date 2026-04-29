# ViTEff: Vision Transformer (Efficient)
(Work in progress)

Fast ViT training on modern stack (blackwell GPUs with CUDA 13).

## Acceleration
- [x] `torch.compile`
- [x] flash attention
- [x] varlen attention
- [x] fp8 training
- [x] semi-structured sparsity
- [ ] spdl dataloading

## Stability
- [x] QK norm
- [x] LayerScale
- [x] gradient clipping
- [x] exponential moving average

## Tasks
- [x] multiclass classification
- [ ] multilabel classification
- [ ] semantic segmentation

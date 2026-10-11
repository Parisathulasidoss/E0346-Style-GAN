# Results

Experiment: compact StyleGAN-style generator trained on a 20,000-image subset of CIFAR-10 (32×32 RGB) for 10 epochs on CPU (TensorFlow 2.20, Python 3.9, ~25 min total).

## Setup

| Item | Value |
|---|---|
| Training images | 20,000 (first 20k of CIFAR-10 train) |
| Image size | 32×32×3, scaled to [-1, 1] |
| Batch size | 64 |
| Epochs | 10 |
| Optimizer | Adam (lr 2e-4, β1 = 0.0, β2 = 0.99), generator and discriminator |
| Loss | Non-saturating logistic GAN loss |
| Latent / style dim | 128 / 128 |
| Generator parameters | 1,044,651 |
| Discriminator parameters | 428,385 |
| Seed | 42 |

## Training log

| Epoch | D loss | G loss | Elapsed (s) |
|---|---|---|---|
| 1 | 1.399 | 0.713 | 133 |
| 2 | 1.389 | 0.705 | 244 |
| 3 | 1.389 | 0.704 | 335 |
| 4 | 1.389 | 0.704 | 425 |
| 5 | 1.379 | 0.716 | 516 |
| 6 | 1.389 | 0.728 | 942 |
| 7 | 1.388 | 0.705 | 1102 |
| 8 | 1.387 | 0.696 | 1232 |
| 9 | 1.387 | 0.694 | 1355 |
| 10 | 1.387 | 0.694 | 1513 |

![Training losses](results/training_losses.png)

## Figures

**Real training samples**

![Real](results/real_samples.png)

**Generated samples (fixed latent vectors, after 10 epochs)**

![Generated](results/generated_samples.png)

**Style mixing** (A supplies early styles, B supplies later styles)

![Style mixing](results/style_mixing.png)

**Latent interpolation** (z₁ → z₂)

![Interpolation](results/latent_interpolation.png)

## Analysis

- The losses stay almost constant at D ≈ 1.387 and G ≈ 0.694 from epoch 2 onward. These are ≈ 2·ln 2 and ln 2, the values obtained when the discriminator outputs ~0.5 for everything, so the discriminator is not separating real from fake. The flat curves do **not** indicate a well-converged model.
- Generated samples show color blobs and texture patterns but no recognizable objects, and many samples look alike (partial mode collapse / low diversity).
- Style mixing and interpolation run correctly, but because sample diversity is low, the differences between images are subtle.
- This is a teaching implementation with a small model, a data subset, and a short CPU run. It does not reproduce original StyleGAN quality.

## Ideas to improve

- Train longer on a GPU, using the full 50,000 CIFAR-10 images.
- Add a stronger discriminator (e.g. more capacity, minibatch standard deviation, spectral norm).
- Add R1 gradient penalty and an exponential moving average (EMA) of generator weights.
- Use equalized learning rate and a lower learning rate for the mapping network (as in StyleGAN).
- Track FID / Inception Score for quantitative evaluation.

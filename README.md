# StyleGAN-style CIFAR-10 Generator (TensorFlow)

A compact, from-scratch implementation of a StyleGAN-inspired generative model in TensorFlow/Keras, trained on color CIFAR-10 (32×32 RGB). It is a teaching implementation, not a reproduction of the original high-resolution StyleGAN.

## Features

- **Mapping network**: 8 dense layers mapping latent `z` to style space `w`
- **Learned constant** 4×4 input to the synthesis network
- **AdaIN** per-layer styling and **per-pixel noise injection**
- **Convolutional discriminator** with non-saturating logistic GAN loss
- **Style mixing** and **latent interpolation** demos

## Project structure

```
.
├── StyleGAN_CIFAR10_TensorFlow.ipynb   # model, training, visualizations
├── requirements.txt
├── RESULTS.md                          # metrics, figures, analysis
├── results/                            # generated figures
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook StyleGAN_CIFAR10_TensorFlow.ipynb
```

Run all cells in order. CIFAR-10 is downloaded automatically by Keras on first run.

Key settings (edit in the notebook):

| Variable | Default | Meaning |
|---|---|---|
| `TRAIN_LIMIT` | 20000 | number of training images |
| `BATCH` | 64 | batch size |
| `EPOCHS` | 10 | training epochs |
| `Z_DIM` / `W_DIM` | 128 | latent and style dimensions |

Training took about 25 minutes on CPU for the default settings. A GPU is recommended for longer runs.

## Results (summary)

10 epochs on 20,000 images produced blurry color/texture patterns without recognizable objects, and the discriminator stayed near chance level (D ≈ 1.387, G ≈ 0.694). See [RESULTS.md](RESULTS.md) for the full log, figures, and suggested improvements.

![Generated samples](results/generated_samples.png)

## Notes

- Tested with TensorFlow 2.20 (Python 3.9 CPU run). The notebook uses `LeakyReLU(negative_slope=...)`, which needs a recent Keras (TensorFlow ≥ 2.16 recommended).
- Add a `LICENSE` file of your choice before publishing.

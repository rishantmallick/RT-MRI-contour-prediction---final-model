# RT-MRI-contour-prediction---final-model
# Articulatory Contour Prediction using Deep Learning

This repository contains the implementation of deep learning models developed for predicting the missing intermediate articulatory contour from Real-Time MRI (RT-MRI) speech sequences. Separate models are trained for **Contour1**, **Contour2**, and **Contour3**, with Variational Autoencoders (VAEs) used for Contour1 and Contour2, and a CNN-based regression model used for Contour3.

---

# Preprocessing

## 1. Arc-Length Interpolation

The contours extracted from the RT-MRI dataset contain varying numbers of points. To ensure a consistent representation, each contour is resampled to **100 points** using **Arc-Length Interpolation**.

### Procedure

1. Compute the Euclidean distance between every pair of consecutive contour points.
2. Set the first contour point as the reference point with cumulative distance equal to zero.
3. Compute the cumulative arc length along the contour.
4. Normalize the cumulative distances by dividing them by the total contour length.
5. Remove duplicate arc-length values using `np.unique()`.
6. Perform linear interpolation independently for the x and y coordinates.
7. Sample 100 uniformly spaced points along the normalized arc length.

Implementation:

```python
def resample(pts, n=100):
    pts = np.array(pts, dtype=np.float32)

    diffs = np.diff(pts, axis=0)
    arc = np.concatenate([[0], np.cumsum(np.sqrt((diffs**2).sum(1)))])

    arc /= arc[-1]

    _, idx = np.unique(arc, return_index=True)
    arc = arc[idx]
    pts = pts[idx]

    t = np.linspace(0, 1, n)

    fx = interp1d(arc, pts[:,0], kind="linear", fill_value="extrapolate")
    fy = interp1d(arc, pts[:,1], kind="linear", fill_value="extrapolate")

    return np.stack([fx(t), fy(t)], axis=1)
```

---

## 2. Normalization

The normalization statistics are computed by concatenating the **past contour** and **future contour**.

```python
mean = np.mean(np.concatenate([past_pts, future_pts], axis=0))
std  = np.std(np.concatenate([past_pts, future_pts], axis=0))
```

The computed mean and standard deviation are then used to normalize

* Past contour
* Future contour
* Ground-truth middle contour

This ensures that all three contours are represented in the same normalized coordinate space.

---

## 3. Data Augmentation

To improve generalization, the training data is augmented using

* Small random rotations
* Small random translations

Different augmentation parameters are applied independently to the **past frame** and the **future frame**, since consecutive speech frames naturally exhibit different articulatory motions.

---

# Post Processing

## Residual Addition

The models predict the **residual contour** rather than the complete contour.

The predicted contour is reconstructed as

```python
linear = (past_pts + future_pts) / 2

prediction = linear + predicted_residual
```

Residual scaling is set to

```
Residual Scaling = 1
```

for both training and inference.

---

## Denormalization

The predicted contour is converted back to the original coordinate system using the same normalization statistics computed during preprocessing.

```python
prediction = prediction * std + mean
```

---

## Coordinate Reconstruction

The complete reconstruction pipeline is

```
Predicted Residual
        ↓
Residual Addition
        ↓
Normalized Prediction
        ↓
Denormalization
        ↓
Final Contour Coordinates
```

---

# Installation

Install the required packages using

```bash
pip install torch
pip install scipy numpy pandas fastdtw
```

---

# Running the Project

The project is designed to run directly in **Google Colab**.

1. Upload the notebook to Google Colab.
2. Mount Google Drive if the model weights are stored there.
3. Run all cells.
4. Load the pretrained weights.
5. Execute the inference cells.

---

# Input and Output

Both the input and output files are MATLAB `.mat` files.

---

# Model Architectures

## Contour 3

### Architecture

Contour3 uses a CNN-based regression network consisting of

* 2 Convolutional Layers
* Adaptive Average Pooling
* 2-Layer MLP Decoder

The CNN encoder extracts local geometric features from the contours, while adaptive pooling reduces the spatial dimension and preserves the most informative features. The decoder predicts the contour coordinates.

### Hyperparameters

```python
Residual Scaling = 1

Learning Rate   = 2.5e-5
Weight Decay    = 1e-5
Smooth Weight   = 0.01

Epochs          = 100
Batch Size      = 32
```

Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2.5e-5,
    weight_decay=1e-5
)
```

Training

* Early Stopping Patience = 40
* Huber Loss
* Smoothness Loss

---

# Contour 2

## Architecture

Contour2 is modeled using a **Variational Autoencoder (VAE)**.

### Encoder

The encoder receives the concatenated past and future contours.

```python
self.encoder = nn.Sequential(
    nn.Linear(400,256),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(256,128),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(128,64),
    nn.ReLU()
)
```

The encoder predicts

```python
self.fc_mu = nn.Linear(64, latent_dim)
self.fc_logvar = nn.Linear(64, latent_dim)
```

where

* `μ` is the latent mean
* `logσ²` is the latent log variance

---

### Reparameterization

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + eps * std
```

---

### Decoder

```python
self.decoder = nn.Sequential(
    nn.Linear(latent_dim, 200)
)
```

The decoder predicts the contour residual.

The final contour is reconstructed as

```python
prediction = linear_interpolation + predicted_residual
```

---

### Loss Function

Reconstruction Loss

```python
reconstruction_loss = 10 * huber_loss(prediction, target)
```

KL Divergence

```python
kl_loss = -0.5 * torch.mean(
    1 + logvar - mu.pow(2) - logvar.exp()
)
```

Total Loss

```python
loss = reconstruction_loss + kl_weight * kl_loss
```

The KL weight is linearly annealed over the first **20 epochs** until it reaches **0.005**.

The VAE loss is further combined with the smoothness loss during training.

### Hyperparameters

```python
Residual Scaling = 1

Learning Rate  = 5e-4
Weight Decay   = 1e-6
Smooth Weight  = 0.02

Epochs         = 100
Batch Size     = 32
```

Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-6
)
```

Training

* Early Stopping Patience = 40
* Huber Reconstruction Loss
* KL Divergence Loss
* Smoothness Loss

---

# Contour 1

## Architecture

Contour1 also uses a **Variational Autoencoder (VAE)**.

The encoder consists of three fully connected layers that generate the latent mean and latent variance.

The latent vector has

```
Latent Dimension = 64
```

The decoder consists of two fully connected layers that predict the contour residual.

The final contour is reconstructed as

```python
prediction = linear_interpolation + predicted_residual
```

### Hyperparameters

```python
Residual Scaling = 1

Learning Rate  = 2e-4
Weight Decay   = 1e-5
Smooth Weight  = 0.01

Epochs         = 100
Batch Size     = 32
```

Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-5
)
```

Training

* Early Stopping Patience = 40
* Huber Reconstruction Loss
* KL Divergence Loss
* Smoothness Loss

---

# Inference

During testing, stochastic sampling is disabled.

```python
z = mu
prediction = decoder(z)
```

The predicted residual is added to the linear interpolation,

```python
prediction = linear_interpolation + predicted_residual
```

followed by denormalization,

```python
prediction = prediction * std + mean
```

to obtain the final contour coordinates.

---

# Summary

| Component           | Contour1                | Contour2                | Contour3                     |
| ------------------- | ----------------------- | ----------------------- | ---------------------------- |
| Model               | Variational Autoencoder | Variational Autoencoder | CNN + Adaptive Pooling + MLP |
| Residual Prediction | ✓                       | ✓                       | ✓                            |
| Latent Dimension    | 64                      | 64                      | —                            |
| Reconstruction Loss | Huber                   | Huber                   | Huber                        |
| KL Divergence       | ✓                       | ✓                       | ✗                            |
| Smoothness Loss     | ✓                       | ✓                       | ✓                            |
| Optimizer           | Adam                    | Adam                    | Adam                         |
| Epochs              | 100                     | 100                     | 100                          |
| Batch Size          | 32                      | 32                      | 32                           |
| Early Stopping      | 40                      | 40                      | 40                           |


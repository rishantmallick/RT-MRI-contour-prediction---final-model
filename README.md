# Rishant_Submission_on_RTMRI
# RT-MRI Articulatory Contour Prediction

This repository contains the implementation of deep learning models for predicting the missing intermediate articulatory contour from Real-Time MRI (RT-MRI) speech sequences. Given the contours of the previous and future frames, the models estimate the contour corresponding to the intermediate frame using residual learning.

---

# Overview

The objective of this work is to reconstruct the missing articulatory contour between two consecutive RT-MRI frames.

For every sample:

* **Input**

  * Previous contour
  * Future contour

* **Output**

  * Predicted intermediate contour

Instead of predicting the complete contour directly, all models predict the **residual** between the linearly interpolated contour and the ground-truth contour.

The final prediction is computed as

```text
Predicted Contour = Linear Interpolation + Predicted Residual
```

This significantly simplifies the learning problem since linear interpolation already provides a strong baseline.

---

# Repository Structure

```text
.
├── Contour1/
├── Contour2/
├── Contour3/
├── Model_Weights/
├── Sample_Input/
├── Sample_Output/
├── README.md
```

---

# Data Preprocessing

## 1. Arc Length Interpolation

The contour annotations originally contain varying numbers of points. To ensure a fixed-size input for all models, every contour is resampled to **100 equally spaced points** using Arc Length Interpolation.

The preprocessing consists of the following steps:

* Convert contour coordinates into a NumPy array.
* Compute the Euclidean distance between consecutive contour points.
* Compute the cumulative arc length beginning from the first point.
* Normalize the cumulative arc length to the interval [0,1].
* Remove duplicate arc-length values using `np.unique()`.
* Generate 100 uniformly spaced positions along the normalized arc.
* Perform linear interpolation independently for the x and y coordinates.
* Stack the interpolated coordinates to obtain a contour of size **100 × 2**.

### Implementation

```python
def resample(pts, n=100):
    pts = np.array(pts, dtype=np.float32)

    diffs = np.diff(pts, axis=0)
    arc = np.concatenate([[0],
                          np.cumsum(np.sqrt((diffs**2).sum(1)))])

    arc /= arc[-1]

    _, idx = np.unique(arc, return_index=True)

    arc = arc[idx]
    pts = pts[idx]

    t = np.linspace(0, 1, n)

    fx = interp1d(arc, pts[:,0], kind='linear')
    fy = interp1d(arc, pts[:,1], kind='linear')

    return np.stack([fx(t), fy(t)], axis=1)
```

---

## 2. Normalization

The previous and future contours are concatenated to compute a common mean and standard deviation.

The same statistics are then used to normalize

* Previous contour
* Future contour
* Ground-truth contour

This ensures that every contour is represented on the same numerical scale during training.

---

## 3. Data Augmentation

To improve generalization, geometric augmentation is applied during training.

The augmentation consists of

* Small random rotations
* Small random translations

Different random transformations are applied independently to the previous and future frames to better simulate realistic articulatory motion.

---

# Post Processing

After the model predicts the residual, several post-processing steps are performed.

## Residual Addition

The predicted residual is added to the linear interpolation.

```text
Linear = (Past + Future) / 2

Prediction = Linear + Predicted Residual
```

Residual scaling is **not used** during either training or inference.

---

## De-normalization

The predicted contour is converted back to the original coordinate system using the same mean and standard deviation computed during preprocessing.

---

## Coordinate Reconstruction

The reconstructed contour is obtained by

1. Computing linear interpolation
2. Adding the predicted residual
3. De-normalizing the contour

---

# Model Architectures

The repository contains separate architectures for each contour.

---

# Contour 1

## Architecture

* 4-layer Multi-Layer Perceptron (MLP)

The network learns high-dimensional representations of articulatory movement and predicts the residual between the interpolated contour and the ground-truth contour.

### Hyperparameters

| Parameter         |     Value |
| ----------------- | --------: |
| Learning Rate     |  2 × 10⁻⁵ |
| Weight Decay      |  1 × 10⁻⁵ |
| Smoothness Weight |      0.01 |
| Residual Scaling  |      0.05 |
| Batch Size        |        32 |
| Epochs            |       100 |
| Optimizer         |      Adam |
| Early Stopping    | 50 epochs |

### Loss Function

* Huber Loss (δ = 0.1)
* Smoothness Loss

---

# Contour 2

## Architecture

* 3-layer MLP Encoder
* Latent Representation
* 3-layer MLP Decoder
* Skip Connection

The encoder extracts compact features from the previous and future contours.

The decoder reconstructs the residual contour from the latent representation.

The skip connection restores spatial information lost during encoding, improving reconstruction accuracy.

### Hyperparameters

| Parameter         |       Value |
| ----------------- | ----------: |
| Learning Rate     | 1.25 × 10⁻⁵ |
| Weight Decay      |    1 × 10⁻⁶ |
| Smoothness Weight |        0.02 |
| Residual Scaling  |        0.04 |
| Batch Size        |          32 |
| Epochs            |         100 |
| Optimizer         |        Adam |
| Early Stopping    |   25 epochs |

### Loss Function

* Huber Loss (δ = 0.1)
* Smoothness Loss

---

# Contour 3

## Architecture

* 2-layer CNN Encoder
* Adaptive Max Pooling
* 2-layer MLP Decoder

The CNN extracts local geometric information from neighbouring contour points.

Adaptive pooling compresses the learned feature maps while preserving the most informative responses.

The MLP predicts the residual contour.

### Hyperparameters

| Parameter         |      Value |
| ----------------- | ---------: |
| Learning Rate     | 2.5 × 10⁻⁵ |
| Weight Decay      |   1 × 10⁻⁵ |
| Smoothness Weight |       0.01 |
| Residual Scaling  |          1 |
| Batch Size        |         32 |
| Epochs            |        100 |
| Optimizer         |       Adam |
| Early Stopping    |  40 epochs |

### Loss Function

* Huber Loss (δ = 0.1)
* Smoothness Loss

---

# CNN + MLP Theory (Contour 3)

The Contour 3 model combines convolutional layers with a fully connected decoder.

The CNN treats the contour as a one-dimensional sequence and applies sliding convolution kernels to neighbouring contour points. These kernels learn local geometric features such as curvature, local slopes, and contour smoothness. Since convolutional filters share weights across the contour, the learned features are translation invariant and generalize well across different articulatory shapes.

Adaptive pooling compresses the feature representation while retaining the strongest responses produced by the convolutional layers.

The pooled feature vector is passed to a Multi-Layer Perceptron, which predicts the residual between the linearly interpolated contour and the ground-truth contour.

Rather than reconstructing the entire contour, the network learns only the non-linear correction required to transform the linear interpolation into the true articulatory contour.

The model is trained using a combination of

* Huber reconstruction loss
* Smoothness loss

The Huber loss improves point-wise accuracy while remaining robust to outliers, whereas the smoothness loss penalizes abrupt changes between neighbouring contour points, resulting in anatomically plausible contours.

---

# Training Configuration

| Parameter   | Value |
| ----------- | ----: |
| Optimizer   |  Adam |
| Batch Size  |    32 |
| Epochs      |   100 |
| Huber Delta |   0.1 |

---

# Required Packages

Install the required dependencies before running the project.

```bash
pip install torch
pip install scipy
pip install numpy
pip install pandas
pip install fastdtw
```

---

# Running Inference

The inference notebook is designed to run directly in **Google Colab**.

1. Upload or open the notebook in Google Colab.
2. Install the required packages.
3. Download the model weights if required.
4. Run all cells.
5. The predicted contours will be saved as MATLAB (`.mat`) files and visualized as PNG images.

---

# Model Weight Names

The model names used during training have been renamed in this repository.

| Training Name                        | Repository Name     |
| ------------------------------------ | ------------------- |
| `contour1_huberfinal-7158`           | `ri_contour1_model` |
| `contour2_huberfinal-imp89`          | `ri_contour2_model` |
| `contour3_huberfinal-4879loudasdsad` | `ri_contour3_model` |

---

# Input and Output Format

Both the input and output files are MATLAB (`.mat`) files.

Each contour is stored as a **100 × 2** array representing the x and y coordinates of the articulatory contour.

---

# Output

The inference pipeline produces:

* Predicted Contour 1
* Predicted Contour 2
* Predicted Contour 3
* MATLAB output file (`.mat`)
* Visualization of predicted contours (`.png`)

---

# Author

**Rishant Mallick**

This repository contains the implementation developed for articulatory contour prediction from Real-Time MRI speech data using deep learning-based residual prediction models.

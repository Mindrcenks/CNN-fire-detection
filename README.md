# Convolutional Neural Networks – Forest Fire Detection

The objective of this project is to perform supervised binary image classification to detect the presence of forest fire in RGB images.  
It employs a convolutional neural network (CNN) architecture and basic preprocessing to classify images into two classes: **fire** and **no_fire**.

---

## Dataset

The dataset used in this project is **Forest Fire Images** from Kaggle:

- Dataset: Forest Fire Images  
- Link: https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images  
- Author: Mohnish Sai Prasad  

Key characteristics:

- RGB images of forest scenes with and without visible fire.
- Two classes: **Fire** and **Non Fire** (binary classification).
- Images are resized to **100×100×3** (RGB) and normalized before training (this preprocessing is done inside the notebook).
- The dataset is split into training, validation and test subsets in the notebook.

---

## Repository Structure

- `fire_detection.ipynb` – main Jupyter Notebook with data loading, model definition, training and evaluation.
- `forest_fire_model.keras` – trained CNN model saved in Keras format.
- `requirements.txt` – list of Python dependencies required to run the notebook.
- (optional) `examples/` – example images and visual results of the model (if you decide to add them later).

---

## CNN Architecture – Fire Classification

The CNN model is designed for binary classification and consists of convolutional, pooling and fully connected layers.

**Input:**

- RGB image of shape **100×100×3**.

**Feature extraction (convolution + pooling):**

- Block 1:
  - `Conv2D(32, kernel_size=3×3, strides=1, activation='relu')`
  - `MaxPooling2D(pool_size=2×2, strides=2)`  
  Output: `50×50×32`

- Block 2:
  - `Conv2D(64, kernel_size=3×3, activation='relu')`
  - `MaxPooling2D(2×2)`  
  Output: `25×25×64`

- Block 3:
  - `Conv2D(128, kernel_size=3×3, activation='relu')`
  - `MaxPooling2D(2×2)`  
  Output: approximately `12×12×128`

**Classification head:**

- `Flatten` → vector of length `12×12×128 = 18,432`
- `Dense(128, activation='relu')`
- `Dense(2, activation='softmax')` – output neurons for **fire** and **no_fire** classes.

Design choices:

- 3×3 kernels capture local edges, textures and color gradients typical for flames while remaining computationally efficient.
- Max pooling (2×2) after each convolutional block reduces spatial resolution and increases robustness to small shifts and noise.

---

## Model Training

The CNN model is compiled and trained with the following configuration:

- Optimizer: **Adam**
- Loss function: **binary crossentropy**
- Metric: **accuracy**
- Training samples: **3200** images
- Batch size: **48**
- Epochs: **20**

Regularization and callbacks:

- **L2 regularization** on model weights to reduce overfitting.
- **EarlyStopping** – stops training if validation accuracy does not improve for several epochs.
- **ModelCheckpoint** – saves the best model to `forest_fire_model.keras`.

Training results:

- Test accuracy: **≈ 97%**
- Training accuracy grows steadily, while validation accuracy shows only small fluctuations, indicating good generalization.

## Post‑processing – Fire Region Localization

For images classified as **fire**, an additional post‑processing step is used to highlight fire regions:

1. Convert the image from RGB to **HSV** color space.
2. Apply color thresholds:
   - Hue ranges for fire‑like colors:
     - Red: `H ∈ [0, 15]`
     - Orange: `H ∈ [15, 25]`
   - Brightness and saturation thresholds:
     - Value (V) > 180
     - Saturation (S) > 150
3. Apply **Sobel filters** along X and Y axes and threshold gradient magnitude (around 70) to remove uniform regions and keep textured fire areas.
4. Use **morphological operations**:
   - Dilation (kernel 7×7, 3 iterations) to merge close fire regions.
   - Closing (2 iterations) to fill gaps within large areas.
   - Opening (2 iterations) to remove small noise.
5. Find contours and keep regions with area ≥ 200 pixels.
6. Draw bounding boxes around detected regions and label them as `"fire"`.

This pipeline reduces false positives and produces clear visual outputs with fire regions highlighted on the original image.

## Requirements

The project uses the following Python packages (see `requirements.txt`):

```txt
tensorflow==2.15.0
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
opencv-python==4.9.0.80
scikit-learn==1.4.2

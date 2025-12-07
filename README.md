# PCA-Based Digit Visualization
This repository demonstrates how **Principal Component Analysis (PCA)** can be applied to pixel-based handwritten digit images.  
The project focuses on *dimensionality reduction*, *visual interpretation of components*, and understanding *how PCA transforms high-dimensional image data into a meaningful latent space*.

<p align="center">
  <img src="https://img.shields.io/badge/ML-PCA-green">
  <img src="https://img.shields.io/badge/Preprocessing-StandardScaler-blue">
  <img src="https://img.shields.io/badge/Visualization-Matplotlib%20%7C%20Seaborn-orange">
  <img src="https://img.shields.io/badge/Category-Unsupervised_Learning-purple">
  <img src="https://img.shields.io/badge/Language-Python-yellow">
</p>

------

# Overview
This project applies **Principal Component Analysis (PCA)** to the classic **Handwritten Digits dataset**.  
The workflow focuses on:

- Extracting pixel data from an 8×8 grayscale image grid  
- Visualizing sample digits as heatmaps and images  
- Reducing 64-dimensional pixel data to 2D and 3D PCA space  
- Understanding variance explained by principal components  
- Visualizing digit clusters based on PCA compressed features  

PCA is especially powerful in high-dimensional image tasks where each pixel is a feature.

------

# Dataset
- **Name:** Digits Dataset (Scikit-Learn)
- **Source:** [Data set](https://github.com/ali-119/PCA-Based-Digit-Visualization-Dimensionality-Reduction/blob/main/digits.csv)
- **Records:** 1,797 handwritten digits (0–9)
- **Features:**
  - 64 pixel intensity values (8×8 image)  
  - `number_label` (target digit)

Each image is stored as a flattened array of 64 grayscale values.

------

# Project Workflow

## 1) Loading the Dataset
Data is loaded from CSV containing pixel values and the digit label.

- Display dataset info  
- Verify 64 pixel columns  
- Confirm no missing values  

## 2) Pixel Extraction
Created a new DataFrame:

<pre>pixels = df.drop("number_label", axis=1)</pre>

Extracted a single digit representation (first row), converted to NumPy array, and reshaped it into 8×8 grid.

## 3) Visualizing Digits
Three visualizations:

- `matplotlib.imshow` (default colormap)  
- `matplotlib.imshow(cmap='gray')`  
- `seaborn.heatmap` with pixel intensities  

These help interpret the pixel intensities and confirm data structure.

## 4) Scaling Pixel Features
Used:

<pre>StandardScaler()</pre>

Scaling is essential because PCA relies on variance, and raw pixel ranges differ across images.

## 5) PCA with 2 Components
Performed dimensionality reduction:

- `PCA(n_components=2)`
- Projected all digits to 2D PCA space
- Visualized using a color-coded scatter plot (`hue = digit label`)

Digits form **distinct natural clusters**, even with only 2 components.

Explained variance of PC1+PC2:

<pre>~21.59%</pre>

## 6) PCA with 3 Components
Extended the model:

- `PCA(n_components=3)`
- 3D scatter plot using Matplotlib's 3D axis
- Color-coded by digit label

This provides an even clearer separation for some digit classes.

------

# Libraries Used
- numpy  
- pandas  
- seaborn  
- matplotlib  
- scikit-learn (PCA, StandardScaler)

------

# How to Run

## Clone the repository:
[GitHub Repository](https://github.com/ali-119/PCA-Based-Digit-Visualization-Dimensionality-Reduction/tree/main)

## Install dependencies:
```python
pip install -r requirements.txt
```
- requirements.txt → [File](https://github.com/ali-119/PCA-Based-Digit-Visualization-Dimensionality-Reduction/blob/main/requirements.txt)

or directly:
```python
pip install numpy pandas seaborn matplotlib scikit-learn
```
Run the script to generate all visualizations.

------

# Results Summary

## PCA with 2 Components
- Captures **~21.6%** of the variance.
- Despite low variance percentage, digits form recognizable clusters.
- Demonstrates PCA’s ability to compress images while retaining structure.

## PCA with 3 Components
- Better separation of digits in 3D space.
- Useful for interactive visualization and cluster analysis.

## Image Reconstruction Insight
Each digit's raw pixel row can be reshaped back into an 8×8 grid to visually confirm the sample.

------

# Key Takeaways
- PCA can cluster handwritten digits even with limited components  
- 64-dimensional pixel data compresses cleanly into 2D and 3D  
- Variance captured by early components contains meaningful structure  
- PCA is suitable for visualization, preprocessing, and noise reduction  

------

# Conclusion
This project demonstrates how **PCA** transforms high-dimensional pixel data into low-dimensional latent space:

- Meaningful digit clusters emerge even with 2 or 3 components  
- Useful for visualization, feature extraction, and preprocessing for ML models  
- Highlights the power of dimensionality reduction on image datasets  

PCA remains a foundational tool in exploratory data analysis for image-based machine learning tasks.

------

# Author ✍️
**Author:** Ali  
**Field:** Data Science & Machine Learning Student  
**Email:** ali.hz87980@gmail.com  
**GitHub:** [ali-119](https://github.com/ali-119)

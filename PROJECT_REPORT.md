# CAPSTONE PROJECT REPORT

# HandWrite AI: An Intelligent Handwriting Recognition, Analysis and Generation System
---

## Abstract

HandWrite AI is an integrated deep learning experimentation and interactive demonstration system for handwritten digit recognition and analysis. It incorporates early computational neuron models, linear classifiers, multilayer perceptrons, optimization, regularization, representation learning, convolutional neural networks, transfer learning, sequential coordinate prediction, and generative adversarial networks. A Streamlit interface enables users to upload handwriting, inspect predictions, compare measured model performance, reconstruct compressed images, visualize generated samples, and examine stroke-coordinate predictions. This report distinguishes achieved results from intended experiments; quantitative results must be inserted after executing the relevant training scripts.

## Problem Statement

Handwritten digit recognition involves variability in stroke thickness, rotation, contrast, spacing and handwriting style. Different neural architectures offer distinct computational and generalization properties. This capstone evaluates such approaches within a consistent experimental environment and provides interactive educational demonstrations of selected representations and predictions.

## Objectives

1. Implement a binary McCulloch–Pitts threshold neuron and a learnable Perceptron.
2. Train an MNIST multilayer perceptron using backpropagation.
3. Compare SGD, RMSProp and Adam under controlled initial weights.
4. Compare PCA with nonlinear autoencoding for dimensionality reduction.
5. Analyze Dropout and Batch Normalization effects on validation accuracy and generalization gap.
6. Implement a LeNet-5-inspired CNN for digit recognition.
7. Adapt VGG16 pretrained on ImageNet to a separately labeled custom numeral dataset.
8. Predict successive pen coordinate locations using an LSTM and UCI PenDigits.
9. Train an unconditional GAN to generate handwritten digit images.
10. Consolidate demonstrations and experimental outputs in a web interface.

## System Architecture

See `assets/architecture.png` for the application-level architecture. Source is divided into data acquisition, preprocessing, ten independent experiments, persisted experimental outputs and Streamlit presentation. These modules share data conventions but do not constitute a single end-to-end neural network.

## Datasets

**MNIST:** Handwritten digit classification (60,000 provided training images and 10,000 test images). The project partitions 5,000 samples from the original training set as validation, then optionally limits training/test examples for accessibility. MNIST images are 28×28 grayscale.

**Custom numerals:** User-collected image folders with train/validation/test splits. Use writer-disjoint splits where possible and provide class counts for reproducibility. Resized to 96×96 RGB for VGG16 transfer learning.

**UCI PenDigits:** 10,992 records with eight ordered coordinate pairs per digit. The records contain spatially resampled trajectories; the network predicts the next coordinate given a preceding window, **not** time-indexed pen motion.

**Offline demonstration:** Scikit-learn's bundled 8×8 digit dataset, held out 20% for evaluating a logistic regression baseline. Distinct from the MNIST models and not intended as a CNN benchmark.

## Methodology

### Foundations: Neuron and Perceptron

The McCulloch–Pitts module computes binary threshold activations for Boolean gates. The Perceptron trains on MNIST classes 0 and 1 and stores linear coefficients. The former is deterministic; the latter learns a decision boundary from labeled examples.

### MLP and Optimization

A two-hidden-layer MLP (256, 128 ReLU activations) with a 10-neuron Softmax head is trained using sparse categorical crossentropy. Optimizer experiments repeat with shared initialization and fixed training examples across SGD (0.01), RMSProp (0.001) and Adam (0.001). Compare validation curves and held-out test metrics, not just training loss.

### Dimensionality Reduction

PCA compresses 784 flattened pixel features into 32 principal components. A Dense Autoencoder compresses the same input to a 32-dimensional bottleneck and is trained with pixel MSE. Reconstructed images and test MSE are compared. This experiment studies reconstruction, not classification performance.

### Regularization

Train baseline, Dropout-only, BatchNorm-only and combined variants. Record train accuracy, validation accuracy, test accuracy and the train/validation accuracy gap. The gap alone is not proof of generalization: also inspect the absolute validation accuracy and loss.

### CNN Classification

The LeNet-5-inspired network uses padded 32×32 images, two convolution and average pooling stages, a 120/84-neuron Dense classifier, and 10-class Softmax output. Evaluate on held-out MNIST test images and save a confusion matrix.

### Transfer Learning

A frozen ImageNet-pretrained VGG16 convolutional backbone is used with a new 10-class classification head. Train on an independently gathered custom dataset (96×96 RGB) with distinct train, validation and test folders. Record the population of each split and state whether writer identities overlap.

### Sequential Prediction

Train a two-layer LSTM with a window of prior PenDigits (x,y) coordinates to predict the next spatially resampled coordinate. Evaluate MSE and MAE on the official PenDigits test partition. Visualize predictions against actual pen paths.

### GAN Generation

Train a generator with transposed convolutions and a convolutional discriminator with adversarial binary crossentropy loss. Generator output uses Tanh in [-1,1]. Display real generated samples after training and report observed quality rather than assuming the generator has converged.

## Experiment Design

**Reproducibility:** Seed 42; record hardware, environment, software versions and all deviations from default CLI arguments.

**Separation of sets:** Training optimizes parameters, validation monitors/compares configurations, final held-out test assesses generalization. Avoid inspecting the test set as part of hyperparameter selection.

**Primary metrics:** Classification accuracy, crossentropy loss, and confusion matrices; PCA/Autoencoder reconstruction MSE; LSTM MSE/MAE; qualitative GAN sample inspection plus training loss trajectories.

## Results and Discussion

**Status:** Results are not supplied in this source package. Replace the following placeholders after executing the code and collecting `artifacts/*.json`.

| Experiment | Actual measured result | Interpretation |
|---|---|---|
| Binary Perceptron | [Insert held-out accuracy] | [Class balance, mistakes] |
| MLP | [Insert held-out accuracy/loss] | [Generalization] |
| SGD/RMSProp/Adam | [Insert table and learning curves] | [Convergence trade-offs] |
| PCA vs Autoencoder | [Insert test reconstruction MSE] | [Visual fidelity] |
| Dropout / BatchNorm | [Insert four model comparison table] | [Overfitting behavior] |
| LeNet-5 | [Insert test accuracy/confusion matrix] | [Classification patterns] |
| VGG16 | [Insert custom test accuracy] | [Transfer / domain shift] |
| LSTM | [Insert next-coordinate MSE/MAE] | [Stroke approximation] |
| GAN | [Include generated image grid] | [Mode collapse/artifact assessment] |

## Limitations and Future Scope

The deployed neural recognition models need trained model files; the offline baseline is a distinct small dataset. Preprocessing is optimized for a single centered digit rather than multi-character handwriting. VGG16 is computationally expensive for a small digit task. The GAN is unconditional and cannot target a chosen numeral. The PenDigits representation is spatially resampled, not a true evenly sampled temporal series. Extensions include segmenting multi-digit strings, collecting writer-disjoint datasets, calibration studies, conditional GAN generation, and genuinely time-stamped pen recordings.

## Conclusion

HandWrite AI integrates major deep learning methods within a common handwriting application. The code provides implementations and an interface for analyzing recognition, optimization, regularization, compression, sequence estimation and generation. Strong empirical conclusions require complete execution, independently measured test metrics and examination of visual outputs; none are implied by the presence of source code alone.

## References

1. LeCun, Y. et al. *Gradient-Based Learning Applied to Document Recognition* (1998).
2. Kingma, D. P. and Ba, J. *Adam: A Method for Stochastic Optimization* (2014).
3. Ioffe, S. and Szegedy, C. *Batch Normalization* (2015).
4. Srivastava, N. et al. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (2014).
5. Simonyan, K. and Zisserman, A. *Very Deep Convolutional Networks for Large-Scale Image Recognition* (2014).
6. Goodfellow, I. et al. *Generative Adversarial Nets* (2014).
7. TensorFlow / Keras official documentation.
8. UCI Machine Learning Repository, *Pen-Based Recognition of Handwritten Digits*.

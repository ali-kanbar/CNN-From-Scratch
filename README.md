# 🧠 Convolutional Neural Network From Scratch (NumPy)

A complete, end-to-end Convolutional Neural Network (CNN) built entirely from scratch using pure **Python** and **NumPy**.

This project was built to gain a deep, mathematical understanding of deep learning mechanics. Instead of relying on high-level frameworks like PyTorch or TensorFlow, this project manually implements the forward passes, backward passes (backpropagation/chain rule), and matrix transformations required to train a neural network.

## ✨ Features Implemented from Scratch

* **Modular Object-Oriented Architecture:** Every layer inherits from a base `Layer` class, allowing the network to be built dynamically.

* **Core Layers:**
  * `Conv`: Convolutional layer with custom sliding-window matrix multiplication.
  * `ReLU`: Rectified Linear Unit activation function.
  * `Max_Pooling`: 2x2 spatial down-sampling.
  * `Reshape`: Matrix flattening.
  * `Dense`: Fully connected linear layer.
  * `Softmax`: Output probability distribution.

* **Loss Engine:** Categorical Cross-Entropy Loss with custom gradient derivation.

* **Advanced Training Mechanics:**
  * **Data Augmentation:** Real-time spatial shifting (`np.roll`) and Gaussian noise injection to prevent overfitting.
  * **Early Stopping:** Automated patience tracking on a dedicated 10% Validation split.
  * **Learning Rate Decay:** Dynamic step-size reduction for fine-tuning weights in later epochs.

## 📊 The Dataset (Fashion MNIST)

The network is trained and evaluated on the **Fashion MNIST** dataset, classifying 10 different items of clothing. It is significantly more complex than the standard MNIST digits dataset, requiring the network to learn nuanced structural features rather than simple edges.

## 🏗️ Network Architecture

Input Image (28x28) 
  ↓
[ Conv Layer ] (3x3 Filter) -> Outputs (26x26)
  ↓
[ ReLU Activation ] -> Zeroes out negative weights
  ↓
[ Max Pooling ] (2x2 Window) -> Downsamples to (13x13)
  ↓
[ Reshape ] -> Flattens to a 169-dimensional vector
  ↓
[ Dense Layer ] -> Maps 169 features to 10 output classes
  ↓
[ Softmax ] -> Converts outputs to 10% - 100% probabilities

## 🚀 Getting Started

### Prerequisites

You only need basic data science libraries to run this project:

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow

*(Note: TensorFlow is imported **exclusively** to download the Fashion MNIST dataset via `keras.datasets`. It is not used for any network logic).*

### Running the Project

1. Clone the repository:
   git clone https://github.com/ali-kanbar/CNN-From-Scratch.git

2. Open `NumPy_CNN_From_Scratch.ipynb` in Jupyter Notebook, VS Code, or Google Colab.

3. Run all cells sequentially.

## 📈 Visualizations & Results

To prove the network is learning true spatial features, the notebook includes built-in diagnostic visualizations.

### 1. CNN Feature Map Extraction

By intercepting the matrix data immediately after the `Conv` layer, we can physically see the edges and structures the network's filter is focusing on:

<img width="975" height="506" alt="image" src="https://github.com/user-attachments/assets/6d789c55-7038-4b37-ba18-eaa71fb4d5fc" />

### 2. The Confusion Matrix

Evaluated on the unseen 10% Test Split to diagnose exact class confusions (e.g., mistaking a Pullover for a Coat):

<img width="990" height="789" alt="image" src="https://github.com/user-attachments/assets/091e7351-54ef-42fd-ad5a-beab143278a1" />

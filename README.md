# Explainability-of-Deep-Learning-Models

Google Colab Link: https://drive.google.com/file/d/1vyUp4ln7KXEPPNGkpduffjkrX7IUDYJs/view?usp=sharing

**Project Overview:**
This project investigates the application of Explainable Artificial Intelligence (XAI) techniques to convolutional neural networks (CNNs) using real-world image data. The study focuses on improving transparency, trust, and interpretability of deep learning models by analyzing how and why predictions are made, rather than only evaluating predictive accuracy.

A dataset of vehicle advertisement images was used as a realistic testbed, where models were trained to detect the presence of embedded vendor contact information, such as phone numbers, email addresses, or URLs. While the dataset itself is confidential, the methodological framework is fully reproducible.

**Problem Motivation:**
While CNNs achieve high performance in image classification tasks, their black-box nature limits their adoption in high-stakes applications. This project addresses the lack of transparency by integrating CAM and Grad-CAM techniques to visualise model attention and expose biases, supporting more trustworthy and human-centred AI systems.

**Methodology:**
Constructed a real-world image dataset containing vehicle advertisements with and without embedded contact information
Initially explored multiclass classification, then transitioned to binary classification due to overlapping visual patterns and class ambiguity
Applied data augmentation and preprocessing (normalisation, geometric transformations) to improve robustness and generalisation

**Trained and compared three CNN architectures:**
Custom Sequential CNN
ResNet50 (transfer learning)
VGG16 (transfer learning)

**Model Evaluation:**
Models were evaluated using training/validation accuracy and loss metrics, confusion matrices, and test-set predictions.
VGG16 (Binary Classification) achieved the best balance between performance and generalisation
Validation Accuracy: 79%
Test Accuracy: ~83%
Other models showed signs of overfitting or instability under validation

**Explainability & Interpretability:**
To analyse model behaviour beyond accuracy,
Confusion matrices were used to identify class imbalance and prediction bias
CAM and Grad-CAM visualisations highlighted spatial regions influencing predictions
Explainability analysis revealed that the model disproportionately focused on one class, exposing bias not evident from accuracy alone
These findings demonstrate how XAI methods can uncover hidden weaknesses and improve the reliability of deep learning systems.

**Key Contributions:**
Comparative evaluation of CNN architectures under real-world data constraints
Demonstration of how explainability methods expose prediction bias and model limitations
Practical framework for integrating XAI into image classification pipelines
Transferable methodology for trustworthy AI, longitudinal imaging, and clinical decision-support systems

**Tools & Technologies:**
Python
TensorFlow / Keras
NumPy, Pandas
Matplotlib, Seaborn
Google Colab (GPU-based training)

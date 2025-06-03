Identifying Plant Leaf Diseases Using Convolutional Neural Networks (CNN)

direct links to google colab: https://colab.research.google.com/drive/1FXqJ50BquyeMyxlj0e9VlKEr9PCVOm_M?usp=sharing

Overview

This project uses deep learning, Convolutional Neural Networks (CNNs) with transfer learning technique with pre-train VGG16 to be precise to classify and identify diseases in plant leaves from image data. With the rise in demand for efficient agricultural solutions, this model aids farmers and agronomists by providing a low-cost, scalable method for early disease detection, potentially improving crop yield and reducing pesticide overuse.

The model is trained on a labeled dataset of diseased and healthy plant leaf images and learns to classify images into distinct disease categories.

Objectives

* Automatically identify plant leaf diseases using image classification.
* Assist in real-time monitoring and diagnosis in agricultural settings.
* Provide a proof-of-concept for scalable AI-powered plant health diagnostics.

Why Convolutional Neural Networks (CNN)?
CNNs are designed to process pixel data and extract hierarchical visual features, making them particularly effective for image classification tasks. This project leverages CNN's ability to:

* Learn complex patterns from leaf textures, color distortions, and lesion shapes.
* Generalize well to new images after training.
* Work effectively with a moderate amount of training data when using data augmentation.

Dataset

Source: plat_village dataset,tfds

https://arxiv.org/abs/1511.08060

Total Classes: 38 classes including both healthy and diseased plant leaves (e.g., Tomato - Bacterial Spot, Potato - Early Blight, Corn - Common Rust, etc.)

Number of Images: ~54,000 images

Image Dimensions: Resized to 128x128 for training efficiency

Split: 70% training, 15% validation, 15% test

Libraries Used

* NumPy
* Pandas
* Matplotlib
* TensorFlow / Keras
* scikit-learn

Limitations

* Trained only on specific plant species and common diseases in the dataset
* May not generalize well to rare diseases or crops not included in training
* Real-world usage would require image capture consistency (lighting, angle)

Future Enhancements
* Integrate real-time image capture via mobile app
* Expand dataset with local agricultural samples
* Use transfer learning with pretrained models like ResNet or EfficientNet
* Local language support for field deployment in rural areas

Use Case
Precision Agriculture for Smallholder Farmers

In regions where access to agronomists is limited, farmers can use smartphone cameras to capture leaf images. This CNN-based model can be deployed via a mobile app to instantly diagnose common diseases and recommend treatment, potentially reducing crop loss and improving food security.

Benefit: Enables early detection, cost-saving, and precision intervention.

Impact: Helps empower farmers with AI tools without the need for deep technical expertise.

Conclusion
This project shows how CNNs can be applied effectively to agriculture, demonstrating strong performance in classifying plant leaf diseases from raw images. By bridging AI and agriculture, this model serves as a stepping stone for smart farming initiatives and real-time disease diagnosis systems.


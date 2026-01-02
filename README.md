# Glaucoma Classification using VGG16 and Multi-Spectral Analysis
Under the supervision of Niloufar Delfan.

This project implements an automated image preprocessing and deep learning pipeline for the detection of Glaucoma. It explores how different color transformations (RGB, GrayScale, HSV, and CMYK) affect the classification performance of a Convolutional Neural Network.

**NOTE**: This has been adapted from a Kaggle notebook which is why there are import statements at different stages of the code.
The project is designed to work with the Drishti-GS Retina Dataset. Classes: 0 (Normal), 1 (Glaucoma).

## Structure: 
The script automatically handles the directory structure for Kaggle environments, splitting data into Train and Test sets.
## Preprocessing Pipeline
Before feeding images into the neural network, the following steps are applied:
1. Noise Removal: Median Filtering and Gaussian Blurring to reduce retinal image artifacts.
2. Histogram Equalization: Enhances image contrast to make the Optic Nerve Head (ONH) more distinct.
3. Color Transformation: Images are converted into four distinct variants: RGB: Standard color. GrayScale: Luminance information only. HSV: Hue, Saturation, and Value. CYM: Cyan, Yellow, and Magenta (CMYK conversion).
4. Normalization: Pixel values are scaled to a [0, 1] range.
5. Data Augmentation: Training images undergo random cropping, horizontal flipping, rotations, shearing, and color jittering to prevent overfitting.

## Model ArchitectureBase Model: 
VGG16 (Pre-trained on ImageNet).

Customizations:The final fully connected layer is replaced to output 2 classes.

Differential Learning Rates: A very small learning rate ($1 \times 10^{-5}$) is applied to the frozen feature extractor, while a larger rate ($1 \times 10^{-3}$) is used for the new classifier.

Optimizer: Adam with weight decay ($1 \times 10^{-4}$).

Loss Function: Cross-Entropy Loss with Class Weights to handle dataset imbalance.

Scheduler: StepLR (decays learning rate by factor of 0.1 every 10 epochs).Early Stopping: Monitored via validation loss to stop training when the model ceases to improve.

Project StructureThe notebook generates the following output structure in /kaggle/working/:
 ```  
├── Train/
│   ├── rgb/
│   ├── grayscale/
│   ├── hsv/
│   └── cym/
├── Test/
│   └── (same as above)
├── models/
│   └── [variant]/best_model.pth
└── plots/
    └── [variant]/training_val.png
```
## Dependencies
The project requires the following Python libraries:
- torch & torchvision (Deep Learning)
- PIL (Image processing)
- numpy (Numerical operations)
- scikit-learn (Stratified splitting and metrics)
- matplotlib & seaborn (Visualization)

## UsageLoad Data: 
1. Ensure the Drishti-GS dataset is mounted in the /kaggle/input/ directory.
2. Run Preprocessing: Execute the "Data Pre-processing" cells to generate the color-transformed folders.
3. Train: Run the "Deep Model" section. The script will iterate through all 4 color variants, training a separate model for each.
4. Evaluate: The final loop generates a Comprehensive Performance Report including: 
- Accuracy, Sensitivity (Recall), and Specificity.
- F1-Score and Precision.
- Confusion Matrix and ROC-AUC Curves.

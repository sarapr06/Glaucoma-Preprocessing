##Different way to upload image in Kaggle than Collab

import ipywidgets as widgets
from IPython.display import display
from PIL import Image as PILImage
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
import numpy as np
import io

#a greyscale version of image
def grayscale_image(img):
    return img.convert("L")

#change hsv of images
def img_hsv(img):
    return img.convert("HSV")

#change CMYK of images
def cym_img(img):
    return img.convert("CMYK")

def apply_median_filter(image_path):
   filtered_img = image_path.filter(ImageFilter.MedianFilter())
   return filtered_img


def apply_gaussian_filter(image_path, radius=2):
    filtered_img = image_path.filter(ImageFilter.GaussianBlur(radius))
    return filtered_img
 def image_equalizer(image):
    return ImageOps.equalize(image)

def normalize_image(image_path):
    img_array = np.array(image_path).astype(np.float32)
    normalized_array = img_array / 255.0
    normalized_img = Image.fromarray((normalized_array * 255).astype(np.uint8))
    return normalized_img

import os
from PIL import Image
import shutil

# Read input files
input_g_dir = '/kaggle/input/drishtigs-retina-dataset-for-onh-segmentation/Training-20211018T055246Z-001/Training/Images/GLAUCOMA'
input_n_dir = '/kaggle/input/drishtigs-retina-dataset-for-onh-segmentation/Training-20211018T055246Z-001/Training/Images/NORMAL'

drishti_g_files = [f for f in os.listdir(input_g_dir) if f.endswith('.png')]
drishti_n_files = [f for f in os.listdir(input_n_dir) if f.endswith('.png')]

# Create Output folder path
os.makedirs('/kaggle/working/Train/rgb', exist_ok=True)
os.makedirs('/kaggle/working/Train/rgb/0_normal', exist_ok=True)
os.makedirs('/kaggle/working/Train/rgb/1_glaucoma', exist_ok=True)

os.makedirs('/kaggle/working/Train/grayscale', exist_ok=True)
os.makedirs('/kaggle/working/Train/grayscale/0_normal', exist_ok=True)
os.makedirs('/kaggle/working/Train/grayscale/1_glaucoma', exist_ok=True)

os.makedirs('/kaggle/working/Train/hsv', exist_ok=True)
os.makedirs('/kaggle/working/Train/hsv/0_normal', exist_ok=True)
os.makedirs('/kaggle/working/Train/hsv/1_glaucoma', exist_ok=True)

os.makedirs('/kaggle/working/Train/cym', exist_ok=True)
os.makedirs('/kaggle/working/Train/cym/0_normal', exist_ok=True)
os.makedirs('/kaggle/working/Train/cym/1_glaucoma', exist_ok=True)

print ("Start Glaucoma train set Processing")

for image_file in drishti_g_files:
    # Full input path
    img_path = os.path.join(input_g_dir, image_file)
    
    print(image_file)
    
    # Open image & Preprocessed
    img = Image.open(img_path)
    processed_one = apply_median_filter(img)
    RGB_img = image_equalizer(processed_one)
   # Save the rgb image
    output_path = os.path.join('/kaggle/working/Train/rgb/1_glaucoma', image_file)
    Normalized_RGB_img = normalize_image (RGB_img)
    Normalized_RGB_img.save(output_path)

    # Save the CMY image
    CMY_img = cym_img(RGB_img)
    output_path = os.path.join('/kaggle/working/Train/cym/1_glaucoma', image_file)
    Normalized_CMY_img = normalize_image (CMY_img)
    Normalized_CMY_img.save(output_path)

    # Save the HSV image
    HSV_img = img_hsv(RGB_img)
    output_path = os.path.join('/kaggle/working/Train/hsv/1_glaucoma', image_file)
    Normalized_HSV_img = normalize_image (HSV_img)
    Normalized_HSV_img.save(output_path)

    # Save the GS image
    GS_img = grayscale_image(RGB_img)
    output_path = os.path.join('/kaggle/working/Train/grayscale/1_glaucoma', image_file)
    Normalized_GS_img = normalize_image (GS_img)
    Normalized_GS_img.save(output_path)

print ("Finish Glaucoma train set Processing")
print ("Start Normal train set Processing")


for image_file in drishti_n_files:
    # Full input path
    img_path = os.path.join(input_n_dir, image_file)
    
    print(image_file)
    
    # Open image & Preprocessed
    img = Image.open(img_path)
    processed_one = apply_median_filter(img)
    RGB_img = image_equalizer(processed_one)

    # Save the rgb image
    output_path = os.path.join('/kaggle/working/Train/rgb/0_normal', image_file)
    Normalized_RGB_img = normalize_image (RGB_img)
    Normalized_RGB_img.save(output_path)

    # Save the CMY image
    CMY_img = cym_img(RGB_img)
    output_path = os.path.join('/kaggle/working/Train/cym/0_normal', image_file)
    Normalized_CMY_img = normalize_image (CMY_img)
    Normalized_CMY_img.save(output_path)

    # Save the HSV image
    HSV_img = img_hsv(RGB_img)
    output_path = os.path.join('/kaggle/working/Train/hsv/0_normal', image_file)
    Normalized_HSV_img = normalize_image (HSV_img)
    Normalized_HSV_img.save(output_path)

    # Save the GS image
    GS_img = grayscale_image(RGB_img)
    output_path = os.path.join('/kaggle/working/Train/grayscale/0_normal', image_file)
    Normalized_GS_img = normalize_image (GS_img)
    Normalized_GS_img.save(output_path)

print ("Finish Normal train set Processing")

import os
from PIL import Image
import shutil

# Read input files
input_g_dir = '/kaggle/input/drishtigs-retina-dataset-for-onh-segmentation/Test-20211018T060000Z-001/Test/Images/glaucoma'
input_n_dir = '/kaggle/input/drishtigs-retina-dataset-for-onh-segmentation/Test-20211018T060000Z-001/Test/Images/normal'

drishti_g_files = [f for f in os.listdir(input_g_dir) if f.endswith('.png')]
drishti_n_files = [f for f in os.listdir(input_n_dir) if f.endswith('.png')]

# Create Output folder path
os.makedirs('/kaggle/working/Test', exist_ok=True)

os.makedirs('/kaggle/working/Test/rgb', exist_ok=True)
os.makedirs('/kaggle/working/Test/rgb/0_normal', exist_ok=True)
os.makedirs('/kaggle/working/Test/rgb/1_glaucoma', exist_ok=True)

os.makedirs('/kaggle/working/Test/grayscale', exist_ok=True)
os.makedirs('/kaggle/working/Test/grayscale/0_normal', exist_ok=True)
os.makedirs('/kaggle/working/Test/grayscale/1_glaucoma', exist_ok=True)

os.makedirs('/kaggle/working/Test/hsv', exist_ok=True)
os.makedirs('/kaggle/working/Test/hsv/0_normal', exist_ok=True)
os.makedirs('/kaggle/working/Test/hsv/1_glaucoma', exist_ok=True)

os.makedirs('/kaggle/working/Test/cym', exist_ok=True)
os.makedirs('/kaggle/working/Test/cym/0_normal', exist_ok=True)
os.makedirs('/kaggle/working/Test/cym/1_glaucoma', exist_ok=True)

print ("Start Glaucoma test set Processing")

for image_file in drishti_g_files:
    # Full input path
    img_path = os.path.join(input_g_dir, image_file)
    
    print(image_file)
    
    # Open image & Preprocessed
    img = Image.open(img_path)
    processed_one = apply_median_filter(img)
    RGB_img = image_equalizer(processed_one)

    # Save the rgb image
    output_path = os.path.join('/kaggle/working/Test/rgb/1_glaucoma', image_file)
    Normalized_RGB_img = normalize_image (RGB_img)
    Normalized_RGB_img.save(output_path)

    # Save the CMY image
    CMY_img = cym_img(RGB_img)
    output_path = os.path.join('/kaggle/working/Test/cym/1_glaucoma', image_file)
    Normalized_CMY_img = normalize_image (CMY_img)
    Normalized_CMY_img.save(output_path)

    # Save the HSV image
    HSV_img = img_hsv(RGB_img)
    output_path = os.path.join('/kaggle/working/Test/hsv/1_glaucoma', image_file)
    Normalized_HSV_img = normalize_image (HSV_img)
    Normalized_HSV_img.save(output_path)

    # Save the GS image
    GS_img = grayscale_image(RGB_img)
    output_path = os.path.join('/kaggle/working/Test/grayscale/1_glaucoma', image_file)
    Normalized_GS_img = normalize_image (GS_img)
    Normalized_GS_img.save(output_path)

print ("Finish Glaucoma test set Processing")
print ("Start Normal test set Processing")

for image_file in drishti_n_files:
    # Full input path
    img_path = os.path.join(input_n_dir, image_file)
    
    print(image_file)
    
    # Open image & Preprocessed
    img = Image.open(img_path)
    processed_one = apply_median_filter(img)
    RGB_img = image_equalizer(processed_one)

    # Save the rgb image
    output_path = os.path.join('/kaggle/working/Test/rgb/0_normal', image_file)
    Normalized_RGB_img = normalize_image (RGB_img)
    Normalized_RGB_img.save(output_path)

    # Save the CMY image
    CMY_img = cym_img(RGB_img)
    output_path = os.path.join('/kaggle/working/Test/cym/0_normal', image_file)
    Normalized_CMY_img = normalize_image (CMY_img)
    Normalized_CMY_img.save(output_path)

    # Save the HSV image
    HSV_img = img_hsv(RGB_img)
    output_path = os.path.join('/kaggle/working/Test/hsv/0_normal', image_file)
    Normalized_HSV_img = normalize_image (HSV_img)
    Normalized_HSV_img.save(output_path)

    # Save the GS image
    GS_img = grayscale_image(RGB_img)
    output_path = os.path.join('/kaggle/working/Test/grayscale/0_normal', image_file)
    Normalized_GS_img = normalize_image (GS_img)
    Normalized_GS_img.save(output_path)

print ("Finish Normal test set Processing")

!pip install torch torchvision scikit-learn matplotlib seaborn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
import seaborn as sns

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --- 1. Define Data Transformations ---
# Data augmentation for the training set; only normalization for validation and test sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# --- 2. Load and Split the Datasets ---
# IMPORTANT: Update these paths
train_dir = '/kaggle/working/Train/rgb'
test_dir = '/kaggle/working/Test/rgb'

# Load the full training dataset before applying transforms
full_train_dataset = datasets.ImageFolder(train_dir)

# Create a stratified 90/10 split for training and validation
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
indices = list(range(len(full_train_dataset)))
y = full_train_dataset.targets # More direct way to get targets

train_indices, val_indices = next(sss.split(indices, y))

# The custom wrapper class to apply different transforms to subsets of the same dataset
# It inherits from torch.utils.data.Dataset
class DatasetWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        # We need to get the original image and label from the base dataset
        img, label = self.dataset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

# Create subset objects first
train_subset = Subset(full_train_dataset, train_indices)
val_subset = Subset(full_train_dataset, val_indices)

# Now, wrap them with the correct transform
train_dataset = DatasetWrapper(train_subset, transform=data_transforms['train'])
val_dataset = DatasetWrapper(val_subset, transform=data_transforms['val'])

# Load the separate test dataset and apply test transforms
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])


# Create DataLoaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
}
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
class_names = full_train_dataset.classes
print(f"Class names: {class_names}")
print(f"Dataset sizes: {dataset_sizes}")

# --- 3. Handle Class Imbalance (based on training set only) ---
# Calculate class weights from the training split to use in the loss function. [8, 19]
train_class_labels = [full_train_dataset.targets[i] for i in train_indices]
class_counts = np.bincount(train_class_labels)
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)
print(f"Calculated class weights: {class_weights}")

# --- 4. Load the Pre-trained VGG Model and Modify the Classifier ---
# Load a pre-trained VGG16 model. [5, 6]
model_ft = models.vgg16(weights=models.VGG16_Weights.DEFAULT)


# Replace the last fully connected layer. [5]
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)

# --- 5. Define Loss Function, Optimizer, and Learning Rate Scheduler ---
criterion = nn.CrossEntropyLoss(weight=class_weights)

params_to_update = [
    {'params': model_ft.features.parameters(), 'lr': 1e-5},  # Very small learning rate for the expert layers
    {'params': model_ft.classifier.parameters(), 'lr': 1e-3} # Larger learning rate for the new layer
]


optimizer_ft = optim.Adam(params_to_update, lr=1e-3, weight_decay=1e-4)

# Decay LR by a factor of 0.1 every 5 epochs. [11]
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

# --- 6. Training the Model with Early Stopping ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, patience=5):
    """
    Trains a model, monitoring validation loss for early stopping.

    Args:
        model: The PyTorch model to train.
        criterion: The loss function.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        num_epochs (int): The maximum number of epochs to train for.
        patience (int): How many epochs to wait for improvement before stopping.

    Returns:
        A tuple containing:
        - The best model (according to lowest validation loss).
        - A history dictionary of training and validation metrics.
    """
    since = time.time()

    # We need to track the best validation loss and save the model weights that produced it.
    # Lower is better, so we initialize it to infinity.
    best_loss = np.Inf
    best_model_wts = copy.deepcopy(model.state_dict())
    
    epochs_no_improve = 0
    early_stop = False
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        if early_stop:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Step the scheduler only during the training phase
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store metrics for plotting
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.cpu().item())

            # --- THE CORE CHANGE IS HERE ---
            # Check for early stopping based on validation loss
            if phase == 'val':
                if epoch_loss < best_loss:
                    print(f'Validation loss improved from {best_loss:.4f} to {epoch_loss:.4f}. Saving model...')
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0 # Reset the patience counter
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience:
                    early_stop = True

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:.4f}')

    # load best model weights and return it
    model.load_state_dict(best_model_wts)
    return model, history

# First, DEFINE the helper function for evaluation
import torch.nn.functional as F # We need softmax

def evaluate_model(model, dataloader):
    """
    Sets the model to evaluation mode and computes predictions on a dataset.

    Args:
        model: The trained PyTorch model.
        dataloader: The DataLoader for the dataset to be evaluated.

    Returns:
        A tuple containing three lists:
        y_true: The ground truth labels.
        y_pred: The model's final predicted class labels.
        y_scores: The model's output probability scores for the positive class.
    """
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []
    y_scores = [] # To store the probability scores

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            
            # Get final predictions (the class with the highest probability)
            _, preds = torch.max(outputs, 1)
            
            # Get probability scores using softmax
            # We take the probabilities for the 'positive' class (index 1)
            scores = F.softmax(outputs, dim=1)[:, 1]

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(scores.cpu().numpy())
            
    return y_true, y_pred, y_scores
    
# Import the necessary functions from sklearn
from sklearn.metrics import roc_auc_score, roc_curve



image_variants = ['rgb', 'cym', 'hsv', 'grayscale']  # Your 4 folders
base_train_path = '/kaggle/working/Train'
base_test_path = '/kaggle/working/Test'

all_histories = {}
all_accuracies = {}
for variant in image_variants:
    print(f"\n{'='*40}\nProcessing Variant: {variant.upper()}\n{'='*40}")

    # Paths
    train_dir = os.path.join(base_train_path, variant)
    test_dir = os.path.join(base_test_path, variant)

    # Load dataset
    full_train_dataset = datasets.ImageFolder(train_dir)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    indices = list(range(len(full_train_dataset)))
    y = full_train_dataset.targets
    train_indices, val_indices = next(sss.split(indices, y))

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    train_dataset = DatasetWrapper(train_subset, transform=data_transforms['train'])
    val_dataset = DatasetWrapper(val_subset, transform=data_transforms['val'])
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    }
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
    class_names = full_train_dataset.classes
    print(f"Class names: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    # Class weights 
    train_class_labels = [full_train_dataset.targets[i] for i in train_indices]
    class_counts = np.bincount(train_class_labels)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    print(f"Calculated class weights: {class_weights}")

    # Model
    model_ft = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_ft = optim.Adam([
        {'params': model_ft.features.parameters(), 'lr': 1e-5},
        {'params': model_ft.classifier.parameters(), 'lr': 1e-3}
    ], lr=1e-3, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    # --- Train ---
    model_ft, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=200, patience=25)

    # Save model
    # Save model
    model_save_dir = os.path.join('models', variant)
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(model_ft.state_dict(), os.path.join(model_save_dir, 'best_model.pth'))

    # Save plots
    plot_save_dir = os.path.join('plots', variant)
     os.makedirs(plot_save_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Training and Validation Accuracy")
    plt.plot(history['train_acc'], label="Train")
    plt.plot(history['val_acc'], label="Val")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Training and Validation Loss")
    plt.plot(history['train_loss'], label="Train")
    plt.plot(history['val_loss'], label="Val")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, 'training_val.png'))
    plt.show()

    # Evaluate
    y_true, y_pred, y_scores = evaluate_model(model_ft, test_loader)
    print("--- Classification Report  ---")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    auc_score = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # ROC Plot
    plt.figure(figsize=(8, 8))
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (1 - Specificity)'); plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve'); plt.legend(loc="lower right")
    plt.savefig(os.path.join(plot_save_dir, 'roc_curve.png')); plt.show()

    #performance report
    print("\n\n" + "="*50)
    print("---     Final Comprehensive Performance Report     ---")
    print("="*50)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print("-" * 25)
    positive_class_label = class_names[1] # Assumes positive class is index 1
    print(f"Metrics for POSITIVE Class ('{positive_class_label}'):")
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  - Sensitivity (Recall): {sensitivity:.4f}")
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"  - Precision: {precision_pos:.4f}")
    f1_pos = 2 * (precision_pos * sensitivity) / (precision_pos + sensitivity) if (precision_pos + sensitivity) > 0 else 0
    print(f"  - F1-Score: {f1_pos:.4f}")
    negative_class_label = class_names[0] # Assumes negative class is index 0
    print(f"\nMetrics for NEGATIVE Class ('{negative_class_label}'):")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"  - Specificity: {specificity:.4f}")
    print("-" * 25)
    print(f"Area Under ROC Curve (AUC): {auc_score:.4f}")
    print("-" * 25)
    print("\nRaw Confusion Matrix Values:")
    print(f"  - True Positives (TP): {tp}")
    print(f"  - True Negatives (TN): {tn}")
    print(f"  - False Positives (FP): {fp}")
    print(f"  - False Negatives (FN): {fn}")
    print("\n" + "="*50)





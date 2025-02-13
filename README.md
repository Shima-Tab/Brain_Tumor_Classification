# Brain Tumor Classification using Deep Learning

## Overview
This project implements a deep learning-based approach for classifying brain tumors using MRI images. The model is built using **PyTorch** and leverages **ResNet-50** for transfer learning. Key features include:
- **Data Augmentation** for improved generalization
- **Mixed Precision Training** for faster performance
- **Early Stopping** to prevent overfitting
- **Grad-CAM Visualization** for model interpretability
- **Learning Rate Scheduling** for better convergence

## Dataset
The dataset consists of MRI scans categorized into four classes:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

Download and prepare the dataset using `download_dataset.py` before running the main script.

## Project Structure
```
brain_tumor_classification/
│── download_dataset.py     # Script to download and prepare dataset
│── train_model.py          # Main training script
│── best_brain_tumor_model.pth  # Saved best model
│── README.md               # Documentation
│── brain_tumor_dataset/    # Folder containing dataset
```

## Installation
Ensure you have Python 3.8+ installed, then run:
```bash
pip install torch torchvision torchaudio matplotlib numpy pandas opencv-python pillow tqdm pytorch-grad-cam
```

## Training the Model
To train the model, run:
```bash
python train_model.py
```

The model will be trained using **ResNet-50**, **AdamW optimizer**, and **ReduceLROnPlateau scheduler**. The best model is saved automatically as `best_brain_tumor_model.pth`.

## Testing the Model
To evaluate the model on the test dataset:
```bash
python test_model.py
```

## Visualizing Model Predictions
To apply **Grad-CAM** for visualization:
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image

image_path = "path_to_test_image.jpg"
image = preprocess_image(image_path)
target_layers = [model.model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
heatmap = cam(image)
```

## Results
The model achieves an accuracy of **~90%** on the test set.

## License
This project is licensed under the MIT License.


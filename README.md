A U-Net deep learning model for segmenting brain tumors from MRI images. It preprocesses data, trains on tumor masks, and saves the model for efficient and accurate medical image analysis.
# Brain Tumor Segmentation using U-Net

A deep learning project for **automatic segmentation of brain tumors in MRI images**, leveraging the U-Net architecture. The solution processes medical images, applies efficient data handling and normalization, trains a robust U-Net model, and outputs precise tumor region masks for clinical or research use.

## Features

- **End-to-End Segmentation Pipeline:** Loads, preprocesses, and normalizes MRI images and corresponding segmentation masks.
- **U-Net Model:** Implements a reliable U-Net with skip connections, batch normalization, and dropout for strong performance on biomedical image segmentation tasks.
- **Easy Data Handling:** Supports RGB images and single-channel masks, with configurable input size.
- **Train/Validation Split:** Includes automatic dataset partitioning for reliable evaluation.
- **Exportable Model:** Saves the trained model in HDF5 format for reuse or deployment.

## Usage

### 1. Clone the Repository


### 2. Prepare Your Data

- Place MRI images in the `images` folder and corresponding tumor masks in the `masks` folder.
- Update `image_path` and `mask_path` variables in the code if directory structure is different.

### 3. Install Requirements


Ensure you have:
- Python 3.x
- Keras
- TensorFlow
- numpy, scikit-learn, PIL

### 4. Run the Training Script


- The script will:
  - Load and preprocess your data
  - Train the U-Net model
  - Save the model as `optimized_brain_tumor_segmentation.h5`

## Model Details

- **Architecture:** U-Net with encoder-decoder, skip connections, batch normalization, dropout for overfitting control, and upsampling in the decoder.
- **Input:** RGB MRI images, resized and normalized.
- **Output:** Binary mask (grayscale, same size as input) representing tumor regions.
- **Loss:** Binary cross-entropy
- **Optimizer:** Adam


## Applications

- **Assists radiologists:** Accurate tumor delineation in medical imaging workflows.
- **Baseline for research:** Can be extended for multi-class segmentation, advanced loss functions, or improved data augmentation techniques.

## Example Directory Structure

project/
│
├── images/<br>
│ ├── image_1.png<br>
│ ├── image_2.png<br>
│ └── ...<br>
├── masks/<br>
│ ├── mask_1.png<br>
│ ├── mask_2.png<br>
│ └── ...<br>
├── train.py<br>
├── requirements.txt<br>
└── README.md<br>


## References

- Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, arXiv:1505.04597
- Datasets commonly used: [BraTS], [Kaggle Examples]



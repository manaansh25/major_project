# Robust Violence Detection under Degraded Visual Conditions

This project aims to detect violence in video streams under various visual conditions using deep learning techniques. 

## Project Overview
The repository contains pipelines for training a 3D Convolutional Neural Network (R3D_18) on video datasets and running real-time webcam inference using YOLO models.

## Dataset
- **Real Life Violence Situations (RLVS)**
- The dataset is processed by sampling 16 frames per video, resizing them to 112x112, and applying normalization. 
- *Note: Dataset files are not included in this repository due to size and licensing constraints.*

## Architecture
- **Training Model**: `R3D_18` (3D ResNet) from `torchvision` modified for binary classification (Violence vs. Non-Violence).
- **Inference Model**: YOLO (`yolo26m.pt`) is used for live webcam detection.

## Project Structure
- `src/`: Contains source code for data loading (`dataset.py`, `dataloader.py`), model definition (`model.py`), and the training loop (`train.py`).
- `results/`: Directory for storing training results and dataset split CSVs.
- `webcam_test.py`: Script to run live violence detection via webcam using YOLO.
- `requirements.txt`: Python dependencies.

## Setup

```bash
# Install required dependencies
pip install -r requirements.txt
```

## Usage

To train the R3D_18 model:
```bash
cd src
python train.py
```

To run real-time inference using your laptop's webcam:
```bash
python webcam_test.py
```

# STKDec

This repository contains the PyTorch implementation of the paper: **Break “Chicken-Egg”: Cross-city Battery Swapping Demand Prediction via Knowledge-guided Diffusion**, 2024.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)

## Introduction
<p style="text-align: justify;">
We propose STKDec, a knowledge-enhanced conditional diffusion model for cross-city battery swapping demand prediction. STKDec leverages historical data from the city with deployed BSS networks to predict battery swapping demands in the target city. Specifically, it first construct an urban knowledge graph (UKG) to align environmental representations and design a multi-relation-aware GCN to transfer inter-station relationship embeddings between source and target cities. Furthermore, an MLP network is employed to capture and model the users' battery-swapping behavior representations. We input all these obtained embeddings into a diffusion model to guide the denoising process. 
</p>

<div align="center">
  
![image](https://github.com/user-attachments/assets/767c58e5-4835-4181-8f72-40c376277103)
</div>

<div align="center">
  
![image](https://github.com/user-attachments/assets/fa8d58ba-6fd7-489b-bd8f-2b80dc851ff4)
</div>

## Requirements
The codebase is implemented in Python 3.6.6. The required packages are listed below:

```bash
numpy==1.15.1
pytorch==1.0.1
```

```
project-root/
│
├── Data-Preparation.ipynb # Notebook for data preprocessing and preparation
├── Diffusion_Model.py # Main script for training the diffusion model
├── Evaluate_Model.py # Script for evaluating the trained model
├── models/ # Directory containing model definitions
│ ├── diffusion_model.py # Diffusion model implementation
│ └── other_models.py # Additional model implementations
├── data/ # Directory containing data files
│ ├── train_data.csv # Training data file
│ ├── test_data.csv # Testing data file
│ └── preprocessing_config.json # Configuration file for preprocessing
├── results/ # Directory for storing results and evaluation outputs
│ ├── evaluation_report.txt # Evaluation report and metrics
│ └── model_checkpoint.pth # Saved model checkpoints
└── README.md # This file
```

## Usage
1.Prepare your dataset and preprocess it using **Data-Preparation.ipynb**

2.Run the training script:

```bash
python Diffusion_Model.py
```
3.Evaluate its performance:

```bash
python Evaluate.py
```
## Results
<div align="center">
  
![image](https://github.com/user-attachments/assets/d3c32e03-04f8-41cd-bba7-dd6a43625a01)
</div>

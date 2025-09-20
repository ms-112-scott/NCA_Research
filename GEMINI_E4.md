# GEMINI_E4.md

## Project Overview

This project, located in the `E4_PI_NCA` directory, focuses on the development and training of a Physics-Informed Neural Cellular Automata (PI-NCA) for simulating fluid dynamics. The project utilizes PyTorch to build and train the NCA model. The experiments are conducted in both Python scripts and Jupyter Notebooks, allowing for both systematic training and interactive exploration.

## Key Components

### 1. `CFD_PI-NCA.py`

This is the main script for the E4 experiment. It orchestrates the entire workflow, including data loading, model definition, training, and evaluation.

- **Data Loading and Preprocessing:** The script loads CFD data from a `.npz` file, splits it into training, evaluation, and test sets, and creates an epoch pool for training.
- **Model Architecture:** The `CAModel` is a PyTorch `nn.Module` that consists of a perception block and a rule block. The perception block uses a set of fixed convolutional kernels (identity, Sobel, Laplacian) to perceive the local neighborhood of each cell. The rule block is a small convolutional neural network that takes the output of the perception block and computes the update for each cell.
- **Training and Evaluation:** The script defines functions for training and evaluating the model for one epoch. The `run_training` function coordinates the entire training process, including the training loop, learning rate scheduling, and early stopping.
- **Loss Function:** The custom loss function is a combination of several losses:
  - `data_mse_loss`: The mean squared error between the predicted and target data.
  - `obstacle_loss`: A loss that penalizes non-zero values inside obstacles.
  - `fft_loss`: A loss that compares the Fourier transforms of the predicted and target data.
  - `Uvel_loss`: A loss that enforces a physical constraint on the velocity field.
- **Metric Function:** The script defines a metric function to evaluate the accuracy of the model.
- **Main Process:** The main part of the script initializes the model, optimizer, and learning rate scheduler, and then calls the `run_training` function to start the training.
- **Testing:** After training, the script loads the trained model and tests it on the test set.

### 2. `notebooks`

This directory contains several Jupyter Notebooks for different purposes:

- `E4-0_PI-NCA.ipynb`: This notebook is the TensorFlow/Keras version of the `CFD_PI-NCA.py` script. It provides a more interactive way to run the experiment and visualize the results.
- `E4-0_PINN_test.ipynb`: This notebook delves deeper into the concept of Physics-Informed Neural Networks (PINNs). It calculates the pressure gradient force, viscous force, and convective force, and then computes the residual. The goal of this notebook seems to be to train an NCA model that minimizes this residual, thereby learning the physical laws of fluid dynamics.
- Other notebooks in this directory are related to data preprocessing, visualization, and other experiments (e.g., Lattice-Boltzmann methods).

### 3. `utils`

This directory contains utility functions used in the project:

- `helper.py`: This file contains a collection of helper functions for various tasks, such as tensor manipulation, plotting, data splitting, and logging.
- `SamplePool.py`: This file defines a `SamplePool` class for managing and sampling data pools.

## Building and Running

To run this project, you will need a Python environment with the following dependencies installed:

- PyTorch
- NumPy
- Matplotlib
- tqdm
- ipynbname

### Running the Main Script

To run the main experiment, you can execute the `CFD_PI-NCA.py` script:

```bash
python E4_PI_NCA/CFD_PI-NCA.py
```

### Running the Notebooks

To explore the experiments and visualizations interactively, you can run the Jupyter Notebooks in the `E4_PI_NCA/notebooks` directory. Make sure you have Jupyter Notebook or JupyterLab installed.

把 E4*PI_NCA 資料夾中所有.ipynb 的 notebook 依照開發的實驗過程順序重新命名為，並且取名為簡易可以讀懂有意義的敘述，檔案用英文重新命名。E4-{number}*{description}

1.先讀取 GEMINI_E4.md 理解所有 E4_PI_NCA 在做甚麼。
2.E4_PI_NCA 資料夾中所有.ipynb 的 notebooks 中，第一最開頭地方新增一個 md cell 記錄這的筆記本相較上一個 或是 主要的改進部分。

- 用繁體中文描述紀錄，簡要條列式
- 不要修改到檔案既有的其他 cell 程式碼

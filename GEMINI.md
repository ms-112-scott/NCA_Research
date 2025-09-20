# GEMINI.md

## Project Overview

This project explores the use of Neural Cellular Automata (NCA) for simulating fluid dynamics. It appears to be a research project with a series of experiments, each building upon the previous one. The project uses TensorFlow and Keras for building and training the NCA models. The experiments are conducted in Jupyter Notebooks, which allows for interactive development and visualization of the results.

The project also involves pre-processing of Computational Fluid Dynamics (CFD) data using ParaView. A Python script is provided to automate this process, converting `.foam` files into NumPy arrays that can be used for training the NCA models.

## Building and Running

The project is primarily run through Jupyter Notebooks. To run the experiments, you will need a Python environment with the following dependencies installed:

*   TensorFlow
*   Keras
*   NumPy
*   Matplotlib
*   OpenCV
*   MoviePy

You will also need to have ParaView installed to run the data pre-processing script.

### Running the Experiments

1.  **Pre-process the CFD data:**
    *   Make sure you have your CFD case files (in `.foam` format) in the `00_RAW_FloorPLan_CFD_CASE` directory.
    *   Run the `ParaviewCasePreprocess.py` script to convert the `.foam` files into NumPy arrays. The script will save the output in the `dataset/FP_case_tensor` directory.
    *   `python ParaviewCasePreprocess.py`

2.  **Run the Jupyter Notebooks:**
    *   Navigate to the directory of the experiment you want to run (e.g., `E1_basicGNCA/notebooks`).
    *   Open the Jupyter Notebook file (e.g., `E1_2-baseline_NCA.ipynb`).
    *   Run the cells in the notebook to train the NCA model and visualize the results.

## Development Conventions

*   The project is written in Python.
*   The code is organized into modules, with shared utilities in the `core_utils` directory.
*   Each experiment is self-contained in its own directory.
*   The use of Jupyter Notebooks with clear headings and comments is encouraged for documenting the research process.
*   The project uses a consistent naming convention for files and directories.

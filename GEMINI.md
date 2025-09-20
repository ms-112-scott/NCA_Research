# GEMINI.md

## Project Overview

This project explores the use of Neural Cellular Automata (NCA) for simulating fluid dynamics. It appears to be a research project with a series of experiments, each building upon the previous one. The project uses TensorFlow and Keras for building and training the NCA models. The experiments are conducted in Jupyter Notebooks, which allows for interactive development and visualization of the results.

The project also involves pre-processing of Computational Fluid Dynamics (CFD) data using ParaView. A Python script is provided to automate this process, converting `.foam` files into NumPy arrays that can be used for training the NCA models.

## Building and Running

The project is primarily run through Jupyter Notebooks. To run the experiments, you will need a Python environment with the following dependencies installed:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- MoviePy

You will also need to have ParaView installed to run the data pre-processing script.

### Running the Experiments

1.  **Pre-process the CFD data:**

    - Make sure you have your CFD case files (in `.foam` format) in the `00_RAW_FloorPLan_CFD_CASE` directory.
    - Run the `ParaviewCasePreprocess.py` script to convert the `.foam` files into NumPy arrays. The script will save the output in the `dataset/FP_case_tensor` directory.
    - `python ParaviewCasePreprocess.py`

2.  **Run the Jupyter Notebooks:**
    - Navigate to the directory of the experiment you want to run (e.g., `E1_basicGNCA/notebooks`).
    - Open the Jupyter Notebook file (e.g., `E1_2-baseline_NCA.ipynb`).
    - Run the cells in the notebook to train the NCA model and visualize the results.

## Development Conventions

- The project is written in Python.
- The code is organized into modules, with shared utilities in the `core_utils` directory.
- Each experiment is self-contained in its own directory.
- The use of Jupyter Notebooks with clear headings and comments is encouraged for documenting the research process.
- The project uses a consistent naming convention for files and directories.

把 E4*PI_NCA 資料夾中所有.ipynb 的 notebook 依照開發的實驗過程順序重新命名為，並且取名為簡易可以讀懂有意義的敘述，檔案用英文重新命名。E4-{number}*{description}

1.先讀取 GEMINI_E4.md 理解所有 E4_PI_NCA 在做甚麼。
2.E4_PI_NCA 資料夾中所有.ipynb 的 notebooks 中，第一最開頭地方新增一個 md cell 記錄這的筆記本相較上一個 或是 主要的改進部分。

- 用繁體中文描述紀錄，簡要條列式
- 不要修改到檔案既有的其他 cell 程式碼
- 不要破壞其他原本的 ipynb 檔案

# Syllabus-Text Consistency Analysis Model

This project provides a complete pipeline to train a model for analyzing the consistency between a syllabus and a teaching text. It uses the `sentence-transformers` library to fine-tune the `all-MiniLM-L6-v2` model on a synthetically generated dataset.

## Project Structure

```
.
├── .gitignore
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── generate_dataset.py
│   ├── dataset.py
│   ├── train.py
│   └── inference.py
├── requirements.txt
└── README.md
```

- **`data/`**: Holds the training and testing datasets.
- **`src/`**: Contains all the Python source code.
- **`src/generate_dataset.py`**: Script to generate the synthetic `train.csv` and `test.csv` datasets.
- **`src/dataset.py`**: Defines the PyTorch `Dataset` for loading the data.
- **`src/train.py`**: Script for training the model. Due to environment constraints preventing the saving of large model files, this script now also includes the evaluation logic, which runs immediately after training.
- **`src/inference.py`**: Example script to run inference with a trained model.
- **`requirements.txt`**: Lists the necessary Python packages.
- **`.gitignore`**: Specifies files and directories to be ignored by Git.

## How to Use

### 1. Installation

First, install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset

Create the synthetic dataset by running the generation script. This will create `train.csv` and `test.csv` in the `data/` directory.

```bash
python src/generate_dataset.py
```

### 3. Train and Evaluate the Model

To train the model, run the `train.py` script. The script will fine-tune the `all-MiniLM-L6-v2` model and then immediately evaluate it on the test set, printing the performance metrics (Accuracy, Precision, Recall, F1-Score).

```bash
python -m src.train
```
**Note**: The fine-tuned model is not saved to disk due to system limitations in the execution environment. In a local environment, the trained model would be saved to the `output/` directory.

### 4. Run Inference

The `inference.py` script demonstrates how to use a sentence-transformer model to check the consistency between a given syllabus and a teaching text. Since the fine-tuned model cannot be saved, this script uses the base `all-MiniLM-L6-v2` model as an example.

```bash
python -m src.inference
```

The script will output a consistency score between 0 and 1. A higher score indicates greater consistency.

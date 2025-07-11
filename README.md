# American Sign Language (ASL) Detection using Deep Learning

This project aims to build a robust deep learning model for detecting and classifying American Sign Language (ASL) alphabets from images. The solution leverages modern computer vision techniques and TensorFlow/Keras for model development and training.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation & Visualization](#evaluation--visualization)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

ASL is a vital communication tool for the deaf and hard-of-hearing community. This project automates the recognition of ASL alphabets from images, enabling applications in accessibility, education, and assistive technology.

## Features

- Image data loading and preprocessing
- Data augmentation for robust training
- Deep learning model for ASL alphabet classification
- Training and validation pipeline
- Visualization of predictions and training progress

## Project Structure

```
asl_detection_dl/
│
├── data/
│   └── raw/
│       └── asl_alphabet_train/   # Raw ASL alphabet images
│
├── src/
│   ├── config.py                 # Configuration settings
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── model.py                  # Model architecture definition
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation and visualization
│   └── utils.py                  # Utility functions
│
├── notebooks/                    # Jupyter notebooks for experiments
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore file
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/asl_detection_dl.git
   cd asl_detection_dl
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv_asl_detection
   venv_asl_detection\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the dataset:**  
   Download the ASL alphabet dataset and place it in `data/raw/asl_alphabet_train/`.

2. **Run data loading and visualization:**
   ```bash
   python src/data_loader.py
   ```

3. **Train the model:**
   ```bash
   python src/train.py
   ```

4. **Evaluate the model:**
   ```bash
   python src/evaluate.py
   ```

## Dataset

- The project uses the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle.
- Ensure the dataset is extracted to `data/raw/asl_alphabet_train/` with subfolders for each alphabet class.

## Training

- The training pipeline uses data augmentation and validation split for robust model performance.
- Model checkpoints and logs are saved for further analysis.

## Evaluation & Visualization

- The evaluation script provides accuracy metrics and visualizes predictions.
- Training progress and sample predictions are plotted for inspection.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and bug fixes.

## License

This project is licensed under the MIT License.
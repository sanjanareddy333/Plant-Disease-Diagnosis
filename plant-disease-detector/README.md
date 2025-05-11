# Plant Disease Detector

A machine learning project for detecting plant diseases from images and recommending treatments.

## Project Structure

```
plant-disease-detector/
├── data/                  # Raw + preprocessed images
├── models/                # Trained model checkpoints
├── src/
│   ├── train.py           # CNN training code
│   ├── mutual_info.py     # MINE / NWJ implementation
│   ├── recommend.py       # Treatment recommendation engine
│   ├── utils.py           # Helpers (data loading, preprocessing)
├── notebooks/
│   ├── EDA.ipynb          # Dataset exploration
│   ├── MI_visualization.ipynb # MI saliency / interpretability
├── README.md
├── requirements.txt
```

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the training: `python src/train.py`

## Usage

[Instructions on how to use the project will go here]


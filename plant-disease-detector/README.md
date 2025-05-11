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


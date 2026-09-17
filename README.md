# Brain Tumor Segmentation (U-Net)

A modularized deep learning pipeline for automated brain tumor segmentation from MRI scans using a custom U-Net architecture. This project has evolved from a single experimental notebook into a structured repository featuring a command-line interface (CLI) for flexible training and a dedicated sandbox for data exploration.

## Project Structure

```text
brain-tumor-segmentation/
│
├── data/
│   └── kaggle_3m/                     # Dataset directory (MRI images and masks)
├── models/
│   └── best_unet_brain_tumor.keras    # Automatically saved best weights
├── src/                             
│   ├── __init__.py                    # Package initialization
│   ├── data.py                        # Automated data loading and preprocessing pipeline
│   ├── model.py                       # Customized U-Net architecture definition
│   └── metrics.py                     # Custom Dice coefficient and BCE-Dice loss functions
├── notebooks/                       
│   ├── exploration.ipynb              # Data inspection & visual debugging
│   ├── classification.ipynb           # Simple CNN for binary tumor classification
└── train.py                           # Fully configurable CLI training script
```

## Architectural Highlights

1. **Custom U-Net Design:** Engineered with a lightweight, optimized footprint starting with 32 filters, scaling down to a 256-filter bottleneck, and using padding='same' with UpSampling2D to cleanly preserve spatial dimensions without complex feature map cropping.

2. **Robust Loss Function:** Combines Binary Cross-Entropy and Soft Dice Loss (bce_dice_loss) to effectively combat class imbalance between background tissue and tumor regions.

3. **Advanced Callbacks:** Integrated EarlyStopping (patience 5) and ReduceLROnPlateau (factor 0.5, patience 2) to ensure stable generalization and prevent overfitting.

4. **Synchronized Data Augmentation:** Utilizes ImageDataGenerator with a fixed seed (666) to ensure parallel, synchronized spatial transformations across input MRIs and ground-truth masks.

## Requirements

- Python 3.11.x
- pip >= 23.x
- Recommended: NVIDIA GPU with CUDA support (optional)

The project is tested with Python 3.11.9.

## Quick Start
1. **Clone the repository:**
```
git clone https://github.com/cheez143/brain-tumor-segmentation.git
cd brain-tumor-segmentation
```

2. **Set up virtual environment**

Make sure Python 3.11 is installed.

Check your version:

*Windows:*
```
py --version
```

*macOS/Linux:*
```
python3.11 --version
```

Create and activate the virtual environment:

*Windows:*
```
py -3.11 -m venv venv
.\venv\Scripts\activate
```

*macOS/Linux:*
```
python3.11 -m venv venv
source ./venv/bin/activate
```

3. **Install dependencies:**

```
pip install -r requirements.txt
```

5. **Prepare Data**

See [Data Documentation & Attribution](data/README.md) for instructions on where to download and how to organize the dataset.

6. **Training via Command-Line Interface (CLI)**

You can train the model directly from your terminal with customizable hyperparameters without modifying any source code. 

*Example:*
```
python train.py --epochs 40 --batch_size 32 --lr 0.0005 --seed 100
```

*Available CLI Flags*

| Flag | Description | Default Value |
| :--- | :--- | :--- |
| `--epochs` | Number of training epochs | `50` |
| `--batch_size` | Batch size for training generator | `16` |
| `--lr` | Initial learning rate for the Adam optimizer | `0.0001` |
| `--seed` | Random seed for data generator synchronization | `666` |
| `--data_dir` | Path to the dataset folder | `data/kaggle_3m/` |

7. **Jupyter Notebook**

We provide interactive Jupyter Notebooks to guide you through the project:
- `notebooks/exploration.ipynb`: Use this to inspect the dataset and understand the metadata distribution.
- `notebooks/classification.ipynb`: An introduction to the data using a simple CNN for binary tumor classification (detecting if a tumor exists).
- Ensure you have installed the requirements, then run:
  ```
  jupyter notebook notebooks/
  ```
## License
This code is released under the **[MIT License](LICENSE)**.

# Brain Tumor Segmentation (U-Net)

A deep learning project implementing a U-Net architecture for pixel-level semantic segmentation of brain tumor (low-grade glioma) from MRI scans.

## Project Overview
This repository provides a complete pipeline to train a U-Net model on medical imaging data. It is designed to be cross-platform, ensuring consistent performance across Windows, macOS, and Linux by using hardware-agnostic code and virtual environments.

## Quick Start
1. **Clone the repository:**
```
git clone https://github.com/cheez143/brain-tumor-segmentation.git
cd brain-tumor-segmentation
```

2. **Set up virtual environment**

*Windows:*
```
python -m venv venv
.\venv\Scripts\activate
```

*macOS/Linux:*
```
python3 -m venv venv
source ./venv/bin/activate
```

3. **Install dependencies:**
```pip install -r requirements.txt```

5. **Prepare Data:**
   See [Data Documentation & Attribution](data/README.md) for instructions on where to download and how to organize the dataset.

6. **Jupyter Notebook**
We provide an interactive Jupyter Notebook for data exploration and visualization.
- `exploration.ipynb`: Use this to inspect the dataset and understand the metadata distribution.
- Ensure you have installed the requirements, then run:
  ```jupyter notebook notebooks/exploration.ipynb```

7. **Train:**
`python train.py --epochs 50 --batch_size 16`

## Contributing
1. **Branching:** Please create a new branch for each new feature (e.g., `git checkout -b feature/model-enhancement`).
2. **Data:** Do not commit data to the repository. Please keep the `data/` directory local and refer to the [data/README.md](data/README.md) for setup instructions.
3. **Pull Requests:** Open a Pull Request once your feature is complete and tested locally.

## License
This code is released under the **[MIT License](LICENSE)**.

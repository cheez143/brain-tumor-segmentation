# LGG Segmentation Dataset

This dataset contains brain MR images together with manual FLAIR abnormality segmentation masks.
The images were obtained from The Cancer Imaging Archive (TCIA).
They correspond to 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection with at least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available.
Tumor genomic clusters and patient data is provided in `data.csv` file.


All images are provided in `.tif` format with 3 channels per image.
For 101 cases, 3 sequences are available, i.e. pre-contrast, FLAIR, post-contrast (in this order of channels).
For 9 cases, post-contrast sequence is missing and for 6 cases, pre-contrast sequence is missing.
Missing sequences are replaced with FLAIR sequence to make all images 3-channel.
Masks are binary, 1-channel images.
They segment FLAIR abnormality present in the FLAIR sequence (available for all cases).


The dataset is organized into 110 folders named after case ID that contains information about source institution.
Each folder contains MR images with the following naming convention:

`TCGA_<institution-code>_<patient-id>_<slice-number>.tif`

Corresponding masks have a `_mask` suffix.

# Data Documentation & Attribution

This project utilizes the **LGG Segmentation Dataset** for brain tumor segmentation tasks.

## Dataset Overview
The LGG (Lower-Grade Glioma) Segmentation Dataset consists of FLAIR abnormality segmentation masks for patients with lower-grade gliomas. The data is derived from collections within **The Cancer Imaging Archive (TCIA)** and **The Cancer Genome Atlas (TCGA)**.

## Attribution & Citations
Please cite the original researchers if you use this dataset in any published work:

* **Buda, M., Saha, A., & Mazurowski, M. A.** (2019). "Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm." *Computers in Biology and Medicine*.
* **Mazurowski, M. A., Clark K., Czarnek N. M., Shamsesfandabadi P., Peters K. B., Saha A.** (2017). "Radiogenomics of lower-grade glioma: algorithmically-assessed tumor shape is associated with tumor genomic subtypes and patient outcomes in a multi-institutional study with The Cancer Genome Atlas data." *Journal of Neuro-Oncology*.

## Licensing
This dataset is licensed under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**.
* **Non-Commercial:** The dataset and any models trained exclusively on this data may not be used for commercial purposes.
* **ShareAlike:** Any adaptations or derived works must be distributed under the same license terms as the original.

## Access & Setup
1. **Download:** The dataset is publicly available on Kaggle: [Brain MRI segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation?resource=download&select=kaggle_3m).
2. **Setup:** Download the files and extract them into this `/data` folder.

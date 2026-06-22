
# Data Documentation & Attribution

This project utilizes the **LGG Segmentation Dataset** for brain tumor segmentation tasks.

## Dataset Overview
The dataset contains brain MR images and manual FLAIR abnormality segmentation masks. It includes 110 patients from The Cancer Genome Atlas (TCGA) lower-grade glioma collection.

* **Format:** All images are in `.tif` format.
* **Channels:** Images have 3 channels.
    * For 101 cases: [Pre-contrast, FLAIR, Post-contrast]
    * For cases with missing sequences: The missing sequence is replaced with the FLAIR sequence to maintain 3-channel consistency.
* **Masks:** Binary, 1-channel images representing the FLAIR abnormality.
* **Metadata:** Patient genomic clusters and clinical data are provided in the `data.csv` file.

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

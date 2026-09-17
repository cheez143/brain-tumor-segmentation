from pathlib import Path
import cv2
import numpy as np

def load_and_preprocess_data(data_dir="../data/kaggle_3m/", img_size=128):
    BASE_DIR = Path(data_dir)

    print("Searching for images and masks...")
    mask_paths = sorted(list(BASE_DIR.rglob("*mask*.*")))
    image_paths = sorted([Path(str(m).replace("_mask", "")) for m in mask_paths])

    if len(mask_paths) == 0:
        raise ValueError(f"No masks found in {data_dir}. Check your path!")

    print(f"Found {len(mask_paths)} pairs of picture-mask.")

    # Loads a color MRI image, converts to RGB, resizes, and normalizes
    def process_image(path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size)) / 255.0
        return img

    # Loads a grayscale tumor mask, resizes, and normalizes
    def process_mask(path):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_size, img_size)) / 255.0
        return mask

    # Load data into NumPy arrays
    print("Processing dataset into arrays...")
    X = np.array([process_image(p) for p in image_paths], dtype=np.float32)
    Y = np.array([process_mask(p) for p in mask_paths], dtype=np.float32)

    # Add a channel dimension to masks so shape matches (Batch, Height, Width, Channels)
    Y = Y[..., np.newaxis]

    print(f"Processed X shape (Images): {X.shape}")
    print(f"Processed Y shape (Masks): {Y.shape}")
    
    return X, Y
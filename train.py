import argparse
import keras
from tensorflow as tf
from src.model import build_full_unet
from src.metrics import bce_dice_loss, dice_coefficient
from src.data import load_and_preprocess_data 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train U-Net via CLI")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--data_dir', type=str, default="data/kaggle_3m/")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    X, Y = load_and_preprocess_data(data_dir=args.data_dir, img_size=128)
    
    val_size = int(len(X) * 0.2)
    X_train, Y_train = X[:-val_size], Y[:-val_size]
    X_val, Y_val = X[-val_size:], Y[-val_size:]

    model = build_full_unet(input_shape=(128, 128, 3))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr), 
        loss=bce_dice_loss, 
        metrics=['accuracy', dice_coefficient]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    AUTOTUNE = tf.data.AUTOTUNE

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    val_dataset = val_dataset.batch(args.batch_size).prefetch(AUTOTUNE)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1000, seed=args.seed)
    train_dataset = train_dataset.batch(args.batch_size).prefetch(AUTOTUNE)

    print("Starting training via CLI...")
    model.fit(
        train_dataset,
        steps_per_epoch=len(X_train) // args.batch_size,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=2
    )
    print("Training complete!")

if __name__ == '__main__':
    main()

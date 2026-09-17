import keras
import tensorflow as tf

#Calculates the Dice overlap score between true mask and prediction
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    # Flatten the tensors to 1D vectors
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    
    # 2 * intersection divided by the sum of all elements + a tiny smooth factor to avoid division by zero
    res = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    return res

#Dice Loss is 1 minus the Dice Coefficient
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

#Combines Binary Cross-Entropy and Dice Loss
def bce_dice_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    
    return bce + dice

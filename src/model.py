from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model

def build_full_unet(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)

    # --- ENCODER (Downsampling) ---
    # Level 1
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1) # 128 -> 64

    # Level 2
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2) # 64 -> 32

    # Level 3
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3) # 32 -> 16

    # --- BOTTLENECK (Deepest Point) ---
    bn = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    bn = Conv2D(256, (3, 3), activation='relu', padding='same')(bn)

    # --- DECODER (Upsampling & Skip Connection) ---
    # Level 3 
    u3 = UpSampling2D((2, 2))(bn) # 16 -> 32
    u3 = concatenate([u3, c3])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    # Level 2 
    u2 = UpSampling2D((2, 2))(c4) # 32 -> 64
    u2 = concatenate([u2, c2])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    # Level 1
    u1 = UpSampling2D((2, 2))(c5) # 64 -> 128
    u1 = concatenate([u1, c1])
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(c6)

    # --- OUTPUT LAYER ---
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c6)

    return Model(inputs=inputs, outputs=outputs)
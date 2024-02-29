from tensorflow import keras

IMAGE_X = 48
IMAGE_Y = 48

def get_cnn_tuning_model(hp):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D( 
        filters=hp.Int("conv_1_filter", min_value=16, max_value=32, step=16),
        kernel_size=hp.Choice("conv_1_kernel", values=[4, 6]),
        activation='relu', input_shape=(IMAGE_X, IMAGE_Y, 1))
    )
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (4, 4), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=80, max_value=182, step=16),
        activation='relu')
    )
    model.add(keras.layers.Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(keras.layers.Dense(37, activation="softmax"))

    model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
    
    return model

def get_cnn_pattern_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (5, 5),
        activation='relu', input_shape=(IMAGE_X, IMAGE_Y, 1))
    )
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (4, 4), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        156,
        activation='relu')
    )
    model.add(keras.layers.Dropout(rate=0.15))
    model.add(keras.layers.Dense(37, activation="softmax"))

    model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
    
    return model
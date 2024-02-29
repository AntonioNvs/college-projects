import numpy as np
import os, sys, math
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

import keras_tuner as kt
from CNN.model import get_cnn_tuning_model, get_cnn_pattern_model
from utils import get_labels, label_indexed, distribution_of_characters, test_random_image
from CNN.model import get_cnn_pattern_model
from utils import test_all_images,plot_accuracy,get_labels, label_indexed, distribution_of_characters, test_random_image, only_test_images

from PIL import Image

def read_image(image_path):
    image = Image.open(image_path).convert("L")

    image = image.resize((image.size[0] + 12, image.size[1] - 2))

    image = np.array(image).astype(np.float32)

    return image

def divide_image(image):
    size_part = math.ceil(image.shape[1] / 6)

    parts = []
    for i in range(0, image.shape[1], size_part):
        if i == 0:
            parts.append(image[:,i:i+size_part+16])
        elif i == image.shape[1] - size_part:
            parts.append(image[:,i-16:i+size_part])
        else:
            parts.append(image[:,i-8:i+size_part+8])

    return parts

def divide_image_without(image):
    size_part = math.ceil(image.shape[1] / 6)

    parts = []
    for i in range(0, image.shape[1], size_part):
        if i == 0:
            parts.append(image[:,i:i+size_part])
        elif i == image.shape[1] - size_part:
            parts.append(image[:,i:i+size_part])
        else:
            parts.append(image[:,i:i+size_part])

    return parts

def get_dataset(folder, title):
    print(f"Reading images of {title}")
    
    if os.path.exists(f"cache/X_{title}.npy") and os.path.exists(f"cache/y_{title}.npy"):
        X = np.load(f"cache/X_{title}.npy")
        y = np.load(f"cache/y_{title}.npy")

        return X, y
    
    filenames = sorted(list(os.listdir(folder)))

    labels = get_labels()

    X = []
    y = []

    for filename in filenames:
        img = read_image(f"{folder}/{filename}")
      
        parts = divide_image(img)

        label = filename.replace(".jpg", "")
        captcha = labels[label]

        for i, part in enumerate(parts):
            X.append(part)
            y.append(captcha[i])

    X = np.array(X) / 255.0
    y = np.array(y)

    y = np.array([label_indexed[v] for v in y])

    np.save(f"cache/X_{title}.npy", X)
    np.save(f"cache/y_{title}.npy", y)

    return X, y


if __name__ == "__main__":
    X_train, y_train = get_dataset("CAPTCHA-10k/treinamento", "train")
    X_validation, y_validation = get_dataset("CAPTCHA-10k/validacao", "validation")
    X_test, y_test = get_dataset("CAPTCHA-10k/teste", "test")

    weights_path = "cache/weights.h5"

    if True:
        # TUNING HYPERPARAMETERS CNN WITH KERAS TUNER
        # tuner = kt.RandomSearch(
        #     get_cnn_tuning_model,
        #     objective="val_accuracy",
        #     max_trials=5
        # )

        # X_train_sample = X_train[:len(X_train) // 5]
        # y_train_sample = y_train[:len(y_train) // 5]
        # X_validation_sample = X_validation[:len(X_validation) // 5]
        # y_validation_sample = y_validation[:len(y_validation) // 5]

        # tuner.search(X_train_sample, y_train_sample, epochs=5, validation_data=(X_validation_sample, y_validation_sample))

        # model = tuner.get_best_models(num_models=1)[0]
        model = get_cnn_pattern_model()

        history = model.fit(X_train, y_train, batch_size=128,
                    epochs=5, validation_data=(X_validation, y_validation))

        test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

        # model.save(weights_path)
    else:
        model = get_cnn_pattern_model()

        model.load_weights(weights_path)

    print("Accuracy: %0.4f  [%s]" % (model.evaluate(X_train, y_train)[1], 'Train set accuracy '))
    print("Accuracy: %0.4f  [%s]" % (model.evaluate(X_test, y_test)[1], 'Test set accuracy '))

    predictions = model.predict(X_test)

    # distribution_of_characters([np.argmax(pred) for pred in predictions])

    # test_random_image(X_test, model)
    # ans = only_test_images(X_test, model)
    # print(ans)
    # plot_accuracy(ans)
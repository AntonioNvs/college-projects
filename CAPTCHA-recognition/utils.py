import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from random import randint

label_indexed = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "?": 10,
    "A": 11,
    "B": 12,
    "C": 13,
    "D": 14,
    "E": 15,
    "F": 16,
    "G": 17,
    "H": 18,
    "I": 19,
    "J": 20,
    "K": 21,
    "L": 22,
    "M": 23,
    "N": 24,
    "O": 25,
    "P": 26,
    "Q": 27,
    "R": 28,
    "S": 29,
    "T": 30,
    "U": 31,
    "V": 32,
    "W": 33,
    "X": 34,
    "Y": 35,
    "Z": 36
}

def get_labels():
    labels = {}
    with open("labels.csv", "r") as src:
        for line in src.read().split("\n"):
            if len(line) < 1: continue 
            key, value = line.split(",")
            labels[key] = value.strip()

    return labels

def test_random_image(X_test, model=None, hog=False):
    if hog:
        predictions = np.array(X_test.mode(axis=1, dropna=True)[0],dtype=np.int32)
    else:
        predictions = model.predict(X_test)
    labels = get_labels()
    index = randint(0, X_test.shape[0] // 6)*6

    i = str(index//6 + 9001)
    text = "0"*(6-len(i)) + i

    inverse = {v: k for k, v in label_indexed.items()}

    print(f"Teste da imagem {i}\n")
    print("Predizido: ", end="")
    for i in range(6):
        if hog:
            print(inverse[predictions[i+index]], end="")
        else:
            print(inverse[np.argmax(predictions[i+index])], end="")
    print(f"\nOriginal: {labels[text]}")

    plt.imshow(mpimg.imread(f"CAPTCHA-10k/teste/{text}.jpg"))
    plt.show()

def test_all_images(X_test, model=None, hog=False):
    if hog:
        predictions = np.array(X_test.mode(axis=1, dropna=True)[0],dtype=np.int32)
    else:
        predictions = model.predict(X_test)
    labels = get_labels()

    inverse = {v: k for k, v in label_indexed.items()}
    dict_accuracy = {
        "0": 0, # "0" means "0/6 correct
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0
    }
    for i in range(X_test.shape[0] // 6):
        index = i*6
        text = "0"*(6-len(str(i+1))) + str(i+1)
        accuracy = 0
        for j in range(6):
            if hog:
                accuracy += 1 if predictions[j+index] == label_indexed[labels[text][j]] else 0
            else:
                accuracy += 1 if np.argmax(predictions[j+index]) == label_indexed[labels[text][j]] else 0
        dict_accuracy[str(accuracy)] += 1
    
    for key, _ in dict_accuracy.items():
       dict_accuracy[key] = sum(x for v,x in dict_accuracy.items() if v >= key)
    return dict_accuracy

def only_test_images(X_test, model=None, hog=False):
    if hog:
        predictions = np.array(X_test.mode(axis=1, dropna=True)[0],dtype=np.int32)
    else:
        predictions = model.predict(X_test)
    labels = get_labels()

    inverse = {v: k for k, v in label_indexed.items()}
    dict_accuracy = {
        "0": 0, # "0" means "0/6 correct
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0
    }

    for i in range(X_test.shape[0] // 6):
        index = i*6
        text = "0"*(6-len(str(i+9001))) + str(i+9001)
        accuracy = 0
        for i in range(6):
            if hog:
                accuracy += 1 if predictions[i+index] == label_indexed[labels[text][i]] else 0
            else:
                accuracy += 1 if np.argmax(predictions[i+index]) == label_indexed[labels[text][i]] else 0
        dict_accuracy[str(accuracy)] += 1

    for key, _ in dict_accuracy.items():
        dict_accuracy[key] = sum(x for v,x in dict_accuracy.items() if v >= key)
    return dict_accuracy


def ploot_accuracy(dict_accuracy):
    keys = list(map(int, dict_accuracy.keys()))
    values = list(dict_accuracy.values())
    values = [v/values[0] for v in values]

    plt.plot(keys, values, marker='o', linestyle='-', color='b')
    plt.xlabel('Numero minimos de caracteres reconhecidos por captcha')
    plt.ylabel('Taxa de acertos')
    plt.title('Resultado')
    # plt.grid(True)
    plt.ylim(0, 1.1)

def distribution_of_characters(characters):
    inverse = {v: k for k, v in label_indexed.items()}

    arr = np.array([inverse[c] for c in characters])

    counter = {value: np.sum(arr == value) / len(arr) for value in set(arr)}

    sorted_keys = sorted(counter.keys())

    counter = {key: counter[key] for key in sorted_keys}

    plt.bar(counter.keys(), counter.values())
    plt.title("Distribuição de frequência das labels")
    plt.xlabel("Labels")
    plt.ylabel("Frequência")
    plt.show()

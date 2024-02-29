import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df_train = pd.read_csv("predictions_train_new.csv", index_col=[0])
df_test = pd.read_csv("predictions_test_new.csv", index_col=[0])
df_validation = pd.read_csv("predictions_validation_new.csv", index_col=[0])

filtering = df_test["MLP"] != df_test["label"]
df_without_mlp = df_test[filtering].drop("MLP", axis=1)

for col in df_without_mlp.columns:                                                                         
    if "label" == col:
        continue
    print(f'Rate {col}: {df_without_mlp[df_without_mlp[col] == df_without_mlp["label"]].shape[0] / df_without_mlp.shape[0]}')    


X_train = df_train.drop("label", axis=1)
y_train = df_train["label"].values

X_test = df_test.drop("label", axis=1)
y_test = df_test["label"].values

X_validation = df_validation.drop("label", axis=1)
y_validation = df_validation["label"].values

def ensemble_model_by_mlp(X_train, X_test, X_validation, y_train, y_test, y_validation):
    X_one_hot = []
    for column in X_train.columns:
        X_one_hot.append(tf.keras.utils.to_categorical(X_train[column], num_classes=37))

    X_train_h = np.concatenate(X_one_hot, axis=1)

    X_one_hot = []
    for column in X_test.columns:
        X_one_hot.append(tf.keras.utils.to_categorical(X_test[column], num_classes=37))

    X_test_h = np.concatenate(X_one_hot, axis=1)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="relu", input_shape=(X_train_h.shape[1], )),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(100, activation="sigmoid"),
        tf.keras.layers.Dense(37, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_train_h, y_train, epochs=50, steps_per_epoch=8, batch_size=12, validation_data=(X_test_h, y_test))

    print(model.evaluate(X_test_h, y_test))


def ensemble_model_by_random_forest(X_train, X_test, X_validation, y_train, y_test, y_validation):
    # Concatenate the training and validation data
    X_train = pd.concat([X_train, X_validation])
    y_train = np.concatenate([y_train, y_validation])

    model = SVC(kernel = 'rbf', random_state = 123, C = 1)
    print("Training SVM..")
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Accuracy: %0.4f  [%s]" % (accuracy_score(y_train, y_train_pred), 'Train set accuracy '))
    print("Accuracy: %0.4f  [%s]" % (accuracy_score(y_test, y_test_pred), 'Test set accuracy '))

ensemble_model_by_mlp(X_train, X_test, X_validation, y_train, y_test, y_validation)
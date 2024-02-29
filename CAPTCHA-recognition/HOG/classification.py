import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.getcwd())

from utils import test_random_image,test_all_images, plot_accuracy, only_test_images
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from tensorflow import keras

def train_test_datasets():
    df_train = pd.read_csv("train.csv", index_col=[0])

    X_train = df_train.drop("0", axis=1)
    y_train = df_train["0"].values

    labels = sorted(set(y_train))
    dict_transform = {label: i for i, label in enumerate(labels)}

    y_train = np.array([dict_transform[v] for v in y_train])

    X_train = (X_train - X_train.mean()) / X_train.std()

    df_test = pd.read_csv("test.csv", index_col=[0])

    X_test = df_test.drop("0", axis=1)
    y_test = df_test["0"].values

    y_test = np.array([dict_transform[v] for v in y_test])

    X_test = (X_test - X_test.mean()) / X_test.std()

    df_validation = pd.read_csv("validation.csv", index_col=[0])

    X_validation = df_validation.drop("0", axis=1)
    y_validation = df_validation["0"].values

    y_validation = np.array([dict_transform[v] for v in y_validation])

    X_validation = (X_validation - X_validation.mean()) / X_validation.std()

    return X_train.values, y_train, X_test.values, y_test, X_validation.values, y_validation

def get_models(X_train, y_train):
    random_state = 20

    mlp = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1], )),
        keras.layers.Dense(156, activation="relu"),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(len(set(y_train)), activation="softmax")
    ])

    mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return {
        "MLP": mlp,
        "SVM": SVC(kernel = 'rbf', random_state = random_state, C = 1),
        "KNN": KNeighborsClassifier(n_neighbors=50, metric='minkowski', p=1),
        "XGB": XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=50, objective='binary:logistic', random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, criterion='gini', random_state = random_state, max_depth=12),
        "Decision Tree": DecisionTreeClassifier(criterion='gini', random_state = random_state, max_depth=12),
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1500),
    }  

def train_models(models, X_train, y_train, X_test, y_test, X_validation, y_validation):

    df_treino = pd.DataFrame(columns=[label for label in models.keys()])
    df_teste = pd.DataFrame(columns=[label for label in models.keys()])
    df_validation = pd.DataFrame(columns=[label for label in models.keys()])

    metrics_data = []

    for label, model in models.items():
        print(f"Training {label}..")

        if label != "MLP":
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_validation_pred = model.predict(X_validation)

            df_treino[label] = y_train_pred
            df_teste[label] = y_test_pred
            df_validation[label] = y_validation_pred

            print("Accuracy: %0.4f  [%s]" % (accuracy_score(y_train,y_train_pred), 'Train set accuracy '))
            print("Accuracy: %0.4f  [%s]" % (accuracy_score(y_validation,y_validation_pred), 'Validation set accuracy '))
            print("Accuracy: %0.4f  [%s]" % (accuracy_score(y_test,y_test_pred), 'Test set accuracy '))
        else:
            model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_validation, y_validation))
            
            train_accuracy = model.evaluate(X_train, y_train)[1]
            test_accuracy = model.evaluate(X_test, y_test)[1]
            validaiton_accuracy = model.evaluate(X_validation, y_validation)[1]

            y_train_pred = [np.argmax(predict) for predict in model.predict(X_train)]
            y_test_pred = [np.argmax(predict) for predict in model.predict(X_test)]
            y_validation_pred = [np.argmax(predict) for predict in model.predict(X_validation)]

            df_treino[label] = y_train_pred
            df_teste[label] = y_test_pred
            df_validation[label] = y_validation_pred

            print("Accuracy: %0.4f  [%s]" % (train_accuracy, 'Train set accuracy '))
            print("Accuracy: %0.4f  [%s]" % (validaiton_accuracy, 'Validation set accuracy '))
            print("Accuracy: %0.4f  [%s]" % (test_accuracy, 'Test set accuracy '))

        metrics_data.append({
            "model": label,
            "accuracy-train": accuracy_score(y_train,y_train_pred),
            "precision-train": precision_score(y_train,y_train_pred, average='macro'),
            "recall-train": recall_score(y_train,y_train_pred, average='macro', zero_division=1),
            "f1-train": f1_score(y_train,y_train_pred, average='macro'),
            "accuracy-test": accuracy_score(y_test,y_test_pred),
            "precision-test": precision_score(y_test,y_test_pred, average='macro'),
            "recall-test": recall_score(y_test,y_test_pred, average='macro', zero_division=1),
            "f1-test": f1_score(y_test,y_test_pred, average='macro'),
            "accuracy-validation": accuracy_score(y_validation,y_validation_pred),
            "precision-validation": precision_score(y_validation,y_validation_pred, average='macro'),
            "recall-validation": recall_score(y_validation,y_validation_pred, average='macro', zero_division=1),
            "f1-validation": f1_score(y_validation,y_validation_pred, average='macro'),
        })

        cm = confusion_matrix(y_test, y_test_pred, labels=list(range(37)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(37)))

        fig, ax = plt.subplots(figsize=(15, 15))
        disp.plot(ax=ax)
        plt.savefig(f"graphs-analysis/confusion_matrix_{label}.png")

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv("metrics.csv", index=False)


    df_treino['label'] = y_train
    df_teste['label'] = y_test
    df_validation['label'] = y_validation
    print('\n')
    return df_treino, df_validation, df_teste

if __name__ == "__main__":
    # X_train, y_train, X_test, y_test, X_validation, y_validation = train_test_datasets()

    # models = get_models(X_train, y_train)

    # df_train, df_validation, df_test = train_models(models, X_train, y_train, X_test, y_test, X_validation, y_validation)

    # df_train.to_csv("predictions_train_new.csv", index=True)
    # df_test.to_csv("predictions_test_new.csv", index=True)
    # df_validation.to_csv("predictions_validation_new.csv", index=True)
    df_train = pd.read_csv("predictions_train_new.csv", index_col=[0])
    df_test = pd.read_csv("predictions_test_new.csv", index_col=[0])
    df_validation = pd.read_csv("predictions_validation_new.csv", index_col=[0])

    new_x = df_train.drop("label", axis=1)
    new_y = df_train["label"].values

    new_x_test = df_test.drop("label", axis=1)
    new_y_test = df_test["label"].values

    new_x_validation = df_validation.drop("label", axis=1)
    new_y_validation = df_validation["label"].values

    train_accuracy= (new_x.mode(axis=1, dropna=True)[0] == new_y).mean()
    validation_accuracy = (new_x_validation.mode(axis=1, dropna=True)[0] == new_y_validation).mean()
    test_accuracy = (new_x_test.mode(axis=1, dropna=True)[0] == new_y_test).mean()

    print("Classificador Ensemble")
    print("Accuracy: %0.4f  [%s]" % (train_accuracy, 'Train set accuracy '))
    print("Accuracy: %0.4f  [%s]" % (validation_accuracy, 'Validation set accuracy '))
    print("Accuracy: %0.4f  [%s]" % (test_accuracy, 'Test set accuracy '))

    #X_merged = pd.DataFrame(np.concatenate((new_x, new_x_validation, new_x_test), axis=0))
    #ans = test_all_images(X_test=X_merged,hog=True)
    test_random_image(X_test=new_x_test,hog=True)
    ans = only_test_images(X_test=new_x_test,hog=True)
    print(ans)
    plot_accuracy(ans)
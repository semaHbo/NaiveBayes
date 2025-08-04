# hazirmodel_test.py

import numpy as np
from model import GNaiveBayes
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

   


    hazir_model = GaussianNB()
    hazir_model.fit(X_train, y_train)
    y_pred_hazir = hazir_model.predict(X_test)
    print("sklearn GaussianNB doğruluk:", accuracy_score(y_test, y_pred_hazir))

if __name__ == "__main__":
    main()

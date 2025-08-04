from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from model import GNaiveBayes
def main():

    iris = load_iris()
    X, y = iris.data, iris.target  

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    #  Modeli eğit
    model = GNaiveBayes()
    model.fit(X_train, y_train)

    # Tahmin yap
    y_pred = model.predict(X_test)

    # Doğruluk
    acc = accuracy_score(y_test, y_pred)
    print(f"Test doğruluğu: {acc:.3f}")

if __name__ == "__main__":
    main()

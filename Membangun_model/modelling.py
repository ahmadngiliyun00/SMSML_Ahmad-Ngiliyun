import os
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data(data_dir: str):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def main():
    # Tracking lokal (sesuai kebutuhanmu)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Student_Academic_Success_Basic")

    data_dir = os.path.join(os.path.dirname(__file__), "namadataset_preprocessing")
    X_train, X_test, y_train, y_test = load_data(data_dir)

    # AUTLOG ONLY (tanpa manual log sama sekali)
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name="basic_autolog_logreg"):
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Output boleh (bukan logging ke MLflow)
        print("Done. test_accuracy =", float(acc))


if __name__ == "__main__":
    main()

# EVOLVE-BLOCK-START

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def preprocess_train_and_predict(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame):
    X_train2 = X_train.dropna()
    y_train2 = y_train.loc[X_train2.index]
    X_val2 = X_val.dropna()
    
    # Ensure labels are DataFrame, then take first column as 1D series
    if isinstance(y_train2, pd.Series):
        y_train2 = y_train2.to_frame()
    y_vec = y_train2.iloc[:, 0]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), selector(dtype_include=["object", "category"])),
            ("num", StandardScaler(with_mean=True), selector(dtype_include=["number"]))
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=500))
    ])

    model.fit(X_train2, y_vec)
    
    # Assume binary labels (e.g., 0/1). Use probability of the positive class (max class).
    proba = model.predict_proba(X_val2)
    y_val2_proba = pd.DataFrame(proba[:, 1], index=X_val2.index, columns=["y_proba"]) 

    return X_val2, y_val2_proba


# EVOLVE-BLOCK-END


def train_and_classify():
    """Trains a classifier on the UCI Adult dataset and evaluates on validation set.

    Returns:
        Tuple (X_val: pd.DataFrame, y_val: pd.DataFrame)
    """
    # Load training and validation data
    X_train = pd.read_csv("X_train.csv", index_col=0)
    y_train = pd.read_csv("y_train.csv", index_col=0)
    X_val = pd.read_csv("X_val.csv", index_col=0)

    X_val_2, y_val2_proba = preprocess_train_and_predict(X_train, y_train, X_val)

    # Return validation features and predictions
    return X_val_2, y_val2_proba

if __name__ == "__main__":
    # Run training/prediction
    X_val_2, y_val2_proba = train_and_classify()

 
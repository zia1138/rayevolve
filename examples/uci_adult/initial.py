# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np

def preprocess_train_and_predict(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering and model training to predict probabilities for the validation set.

    Args:
        X_train: Training features
        y_train: Training labels (0 or 1)
        X_val: Validation features

    Returns:
        A pd.DataFrame with a single column "y_proba" containing predicted probabilities
        for the positive class (1), with the same index as X_val.
    """
    # Simple baseline: Random predictions
    rng = np.random.default_rng()
    proba = rng.random(len(X_val))
    
    return pd.DataFrame(proba, index=X_val.index, columns=["y_proba"])

# EVOLVE-BLOCK-END

# your code here

import pickle
import numpy as np
import pandas as pd


# Load trained models
with open("model_1.pickle", "rb") as f:
    model_1 = pickle.load(f)

with open("model_2.pickle", "rb") as f:
    model_2 = pickle.load(f)

with open("model_3.pickle", "rb") as f:
    model_3 = pickle.load(f)


def roast_category(value):
    """Convert roast names into numerical categories."""

    roast_mapping = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4
    }

    if pd.isna(value):
        return np.nan

    return roast_mapping.get(str(value).strip(), np.nan)


def predict_rating(X, text=False):
    """
    Predict coffee ratings.

    text=False:
        X contains 100g_USD and roast columns.

    text=True:
        X contains a text column with coffee descriptions.
    """

    if text:
        if isinstance(X, pd.DataFrame):
            descriptions = X["text"].fillna("").astype(str)
        else:
            descriptions = pd.Series(X).fillna("").astype(str)

        return model_3.predict(descriptions)

    # Convert roast names into numerical categories
    X = X.copy()
    X["roast_cat"] = X["roast"].apply(roast_category)

    predictions = []

    for _, row in X.iterrows():

        price = row["100g_USD"]
        roast = row["roast_cat"]

        if pd.notna(roast):
            # Use both price and roast
            features = pd.DataFrame(
                [[price, roast]],
                columns=["100g_USD", "roast_cat"]
            )

            prediction = model_2.predict(features)[0]

        else:
            # Unknown roast: use price only
            features = pd.DataFrame(
                [[price]],
                columns=["100g_USD"]
            )

            prediction = model_1.predict(features)[0]

        predictions.append(prediction)

    return np.array(predictions)

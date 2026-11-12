# your code here

import pandas as pd
import numpy as np
import pickle


with open("model_1.pickle", "rb") as f:
    model_1 = pickle.load(f)

with open("model_2.pickle", "rb") as f:
    model_2 = pickle.load(f)


def roast_category(value):

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


def predict_rating(df_X):

    predictions = []

    for _, row in df_X.iterrows():

        price = row["100g_USD"]
        roast = roast_category(row["roast"])

        if pd.isna(roast):
            prediction = model_1.predict(
                pd.DataFrame({"100g_USD": [price]})
            )[0]

        else:
            prediction = model_2.predict(
                pd.DataFrame({
                    "100g_USD": [price],
                    "roast_cat": [roast]
                })
            )[0]

        predictions.append(prediction)

    return np.array(predictions)
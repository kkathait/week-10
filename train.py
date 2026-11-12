
import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================
# Load Dataset
# ==========================================

url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

df = pd.read_csv(url)


# ==========================================
# Exercise 1: Linear Regression
# ==========================================

df_1 = df[["100g_USD", "rating"]].dropna()

X_1 = df_1[["100g_USD"]]
y_1 = df_1["rating"]

lr = LinearRegression()
lr.fit(X_1, y_1)

with open("model_1.pickle", "wb") as f:
    pickle.dump(lr, f)


# ==========================================
# Exercise 2: Decision Tree Regressor
# ==========================================

def roast_category(value):
    """Convert roast categories into numerical values."""

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


# Create numerical roast column
df["roast_cat"] = df["roast"].apply(roast_category)

df_2 = df[["100g_USD", "roast_cat", "rating"]].dropna(
    subset=["100g_USD", "rating"]
)

X_2 = df_2[["100g_USD", "roast_cat"]]
y_2 = df_2["rating"]

dtr = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

dtr.fit(X_2, y_2)

with open("model_2.pickle", "wb") as f:
    pickle.dump(dtr, f)


# ==========================================
# Bonus Exercise 4: TF-IDF Text Regression
# ==========================================

df_3 = df[["desc_3", "rating"]].dropna()

X_3 = df_3["desc_3"].astype(str)
y_3 = df_3["rating"]

model_3 = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("regressor", LinearRegression())
])

model_3.fit(X_3, y_3)

with open("model_3.pickle", "wb") as f:
    pickle.dump(model_3, f)


# ==========================================
# Completion Message
# ==========================================

print("All models successfully trained and saved!")

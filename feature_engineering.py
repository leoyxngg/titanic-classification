import numpy as np


FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
    "HasCabin",
    "Title",
]


def preprocess_data(data):
    data = data.copy()

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = np.where(data["FamilySize"] == 1, 1, 0)
    data["HasCabin"] = np.where(data["Cabin"].isna(), 0, 1)

    data["Title"] = data["Name"].str.extract(r",\s*([^.]*)\.", expand=False)
    common_titles = {"Mr", "Miss", "Mrs", "Master"}
    data["Title"] = np.where(data["Title"].isin(common_titles), data["Title"], "Rare")

    return data

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def preprocess_data(train_data, test_data):
    train_data = train_data.copy()
    test_data = test_data.copy()

    age_median = train_data["Age"].median()
    fare_median = train_data["Fare"].median()
    embarked_mode = train_data["Embarked"].mode()[0]

    for data in (train_data, test_data):
        data.fillna(
            {
                "Age": age_median,
                "Fare": fare_median,
                "Embarked": embarked_mode,
            },
            inplace=True,
        )
        data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
        data["Embarked"] = data["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    return train_data, test_data

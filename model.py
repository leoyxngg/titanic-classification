import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import FEATURES, preprocess_data


HYPERPARAMETER_GRID = {
    "classifier__n_estimators": [150, 300, 600, 900],
    "classifier__max_depth": [4, 6, 8, None],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"],
}

NUMERIC_FEATURES = ["Pclass", "Age", "Fare", "FamilySize", "IsAlone", "HasCabin"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title"]


def main():
    train_data = pd.read_csv("Dataset/train.csv")
    test_data = pd.read_csv("Dataset/test.csv")
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    train_X = train_data[FEATURES]
    train_y = train_data["Survived"]
    test_X = test_data[FEATURES]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        model,
        HYPERPARAMETER_GRID,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )
    grid_search.fit(train_X, train_y)

    print("Best parameters:", grid_search.best_params_)
    print("Best validation score:", grid_search.best_score_)

    predictions = grid_search.predict(test_X)
    submission = pd.DataFrame(
        {
            "PassengerId": test_data["PassengerId"],
            "Survived": predictions,
        }
    )
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()

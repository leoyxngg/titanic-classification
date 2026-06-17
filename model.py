import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import FEATURES, preprocess_data


MODEL_CONFIGS = {
    "1": {
        "name": "Random Forest",
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "classifier__n_estimators": [150, 300, 600, 900],
            "classifier__max_depth": [4, 6, 8, None],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
        },
    },
    "2": {
        "name": "Gradient Boosting Trees",
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "classifier__n_estimators": [150, 300, 600, 900],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__max_depth": [2, 3, 4, None],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
        },
    },
    "3": {
        "name": "XGBoost Trees",
        "estimator": XGBClassifier(random_state=42),
        "param_grid": {
            "classifier__n_estimators": [150, 300, 600, 900],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__max_depth": [2, 3, 4, None],
            "classifier__objective": ["binary:logistic"],
            "classifier__subsample": [0.6, 0.7, 0.9, 1],
            "classifier__colsample_bytree": [0.6, 0.7, 0.9, 1],
            "classifier__eval_metric": ["logloss"],
        }
    }
}

NUMERIC_FEATURES = ["Pclass", "Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone", "HasCabin"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title"]


def choose_model():
    print("Choose a model:")
    for key, config in MODEL_CONFIGS.items():
        print(f"{key}. {config['name']}")

    while True:
        choice = input("Enter model number: ").strip()
        if choice in MODEL_CONFIGS:
            return MODEL_CONFIGS[choice]

        print("Invalid choice. Please enter one of:", ", ".join(MODEL_CONFIGS))


def main():
    model_config = choose_model()

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
            ("classifier", model_config["estimator"]),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Training {model_config['name']}...")

    grid_search = GridSearchCV(
        model,
        model_config["param_grid"],
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
    print("Saved predictions to submission.csv")


if __name__ == "__main__":
    main()

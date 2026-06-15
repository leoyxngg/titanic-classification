import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from feature_engineering import FEATURES, preprocess_data


HYPERPARAMETER_GRID = {
    "n_estimators": [200, 400, 800, 1600],
    "max_depth": [None, 4, 6, 10],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2"],
}


def main():
    train_data = pd.read_csv("Dataset/train.csv")
    test_data = pd.read_csv("Dataset/test.csv")
    train_data, test_data = preprocess_data(train_data, test_data)

    train_X = train_data[FEATURES]
    train_y = train_data["Survived"]
    test_X = test_data[FEATURES]

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        HYPERPARAMETER_GRID,
        scoring="accuracy",
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
"""Baseline Model

Uses a simple, one-layer logistic regression architecture to make basic predictions.
Will be used as a reference for the next model and other future models"""

import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

#Pick out the features we want to learn from in the model
FEATURE_COLUMNS = [
    "artist_total_prior_appearances",
    "artist_festival_prior_appearances",
    "artist_other_festival_prior_appearances",
    "years_since_festival_appearance",
    "festival_previous_year_size",
]
#Set hyperparameters for training split and file paths to save to
DATA_PATH = "data/processed/features.csv"
#Train up until this specific year
TRAIN_END_YEAR = 2024
VALIDATION_YEAR = 2025
TEST_YEAR = 2026
EPOCHS = 300
LEARNING_RATE = 0.01
SEED = 42
MODEL_OUTPUT = "models/logistic_baseline.pt"
METRICS_OUTPUT = "models/logistic_baseline_metrics.json"


class LogisticRegression(nn.Module):
    #Only sets 1 layer of logistic regression
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, inputs):
        return self.linear(inputs).squeeze(1)

#Randomize seed before training
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#Pick out each feature and set up one-hot encoding for each festival
def features(dataframe, means=None, standard_deviations=None):
    numeric = dataframe[FEATURE_COLUMNS].astype(float).copy()
    festival = pd.get_dummies(dataframe["festival"], dtype=float)

    for festival_name in ["Bass Canyon", "Lost Lands"]:
        if festival_name not in festival:
            festival[festival_name] = 0.0
    festival = festival[["Bass Canyon", "Lost Lands"]]
    #Calculate means and std for the inputs
    if means is None:
        means = numeric.mean()
    if standard_deviations is None:
        standard_deviations = numeric.std().replace(0, 1)
    #Standardize features
    numeric = (numeric - means) / standard_deviations
    values = np.concatenate([numeric.to_numpy(), festival.to_numpy()], axis=1)
    return values.astype(np.float32), means, standard_deviations


def calculate_metrics(probabilities, labels):
    #Convert probability values into predictions
    predictions = (probabilities >= 0.5).astype(int)
    labels = labels.astype(int)
    #Set up 4 outcomes of the model
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    #Calculate performance metrics
    precision = true_positive / max(true_positive + false_positive, 1) #percentage of correct predicted appearances
    recall = true_positive / max(true_positive + false_negative, 1) #percentage of actual appearances found
    f1 = 2 * precision * recall / max(precision + recall, 1e-12) #balance of percision and recall
    accuracy = (true_positive + true_negative) / max(len(labels), 1) #percentage of correct predictions
    #PLace results in a dictionary
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }

#evaluate model performance
def evaluate(model, inputs, labels):
    model.eval()
    #turn off gradients for eval
    with torch.no_grad():
        probabilities = torch.sigmoid(model(inputs)).numpy()
    return calculate_metrics(probabilities, labels.numpy())


def train_logistic_model(
    #Set parameters for training
    data_path=DATA_PATH,
    train_end_year=TRAIN_END_YEAR,
    validation_year=VALIDATION_YEAR,
    test_year=TEST_YEAR,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    seed=SEED,
):
    
    set_seed(seed)
    #Feed in the csv into a dataframe
    data = pd.read_csv(data_path)
    #Set training years for data and prepare test & validation years
    train = data[data["year"] <= train_end_year].copy()
    validation = data[data["year"] == validation_year].copy()
    test = data[data["year"] == test_year].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("The requested train, validation, or test split is empty.")
    #Prepare testing, validation, and training inputs
    train_values, means, standard_deviations = features(train)
    validation_values, _, _ = features(
        validation, means, standard_deviations
    )
    test_values, _, _ = features(test, means, standard_deviations)

    train_inputs = torch.tensor(train_values)
    validation_inputs = torch.tensor(validation_values)
    test_inputs = torch.tensor(test_values)
    train_labels = torch.tensor(train["appeared"].to_numpy(), dtype=torch.float32)
    validation_labels = torch.tensor(
        validation["appeared"].to_numpy(), dtype=torch.float32
    )
    test_labels = torch.tensor(test["appeared"].to_numpy(), dtype=torch.float32)
    #Prepare model for training with all values from features call
    model = LogisticRegression(train_inputs.shape[1])
    positive_count = train_labels.sum().item()
    negative_count = len(train_labels) - positive_count
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / max(positive_count, 1))
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #Train the model and use Adam to adjust gradients
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(train_inputs), train_labels)
        loss.backward()
        optimizer.step()
    #Track outputs of the training in a report
    report = {
        "model": "logistic_regression",
        "train_end": train_end_year,
        "validation_year": validation_year,
        "test_year": test_year,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "split_sizes": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "metrics": {
            "train": evaluate(model, train_inputs, train_labels),
            "validation": evaluate(model, validation_inputs, validation_labels),
            "test": evaluate(model, test_inputs, test_labels),
        },
        "numeric_means": means.to_dict(),
        "numeric_standard_deviations": standard_deviations.to_dict(),
    }

    return model, report


def main():
    model, report = train_logistic_model()

    model_path = Path(MODEL_OUTPUT)
    metrics_path = Path(METRICS_OUTPUT)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(FEATURE_COLUMNS) + 2,
            "feature_columns": FEATURE_COLUMNS,
            "numeric_means": report["numeric_means"],
            "numeric_standard_deviations": report[
                "numeric_standard_deviations"
            ],
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

from load_data import load_data
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
import csv

DEVICE = torch.device("cpu")
EPOCHS = 10

x_train, y_train, x_val, y_val, x_test = load_data()

def convert_to_labeled_df(arr):
    return pd.DataFrame(arr, columns=["Region", "Population", "Area", "Pop. Density", "Coastline", "Net migration",
                                    "Infant mortality", "GDP", "Literacy", "Phones", "Arable", "Crops", "Other",
                                    "Birthrate", "Deathrate", "Agriculture", "Industry", "Service",
                                    "Handwashing Facilities", "Extreme poverty", "Median age", "Life expectancy",
                                    "Human development index"])

dfx = convert_to_labeled_df(x_train).drop(columns=["Handwashing Facilities", "Extreme poverty"]).fillna(0)
dfxv = convert_to_labeled_df(x_val).drop(columns=["Handwashing Facilities", "Extreme poverty"]).fillna(0)
dfxt = convert_to_labeled_df(x_test).drop(columns=["Handwashing Facilities", "Extreme poverty"]).fillna(0)

def normalize_by_column(array):
    x = np.array(array, dtype=np.float32)
    norm_array = x / x.max(axis=0)
    return norm_array

# Train Set
inputs = torch.from_numpy(normalize_by_column(dfx.values))
targets = torch.from_numpy(y_train)

# Val Set
np_val_x = torch.from_numpy(normalize_by_column(dfxv.values))
np_val_y = torch.from_numpy(y_val)

# Test Set
np_test_x = torch.from_numpy((normalize_by_column(dfxt.values)))

def define_model():
    layers = [nn.Linear(21, 1)]

    return nn.Sequential(*layers)

def objective(trial):
    model = define_model().to(DEVICE)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # train model
    loss_list = []
    val_loss_list = []
    iteration_number = trial.suggest_int("iterations", 50, 1000)

    for iteration in range(iteration_number):
        # optimization
        optimizer.zero_grad()

        # Forward to get output
        results = model.forward(inputs)

        # Calculate Loss
        loss = mse(results.squeeze(), targets)

        # backward propagation
        loss.backward()

        # Updating parameters
        optimizer.step()

        # store loss
        loss_list.append(loss.data)

        val_results = model.forward(np_val_x)
        val_loss = mse(val_results.squeeze(), np_val_y).data
        val_loss_list.append(val_loss)

        trial.report(val_loss, iteration_number)

    return val_loss_list[-1]





# if __name__ == "__main__":
#     study = optuna.create_study(direction="minimize")
#     study.optimize(objective, n_trials=100, timeout=600)
#
#     pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
#     complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
#
#     print("Study statistics: ")
#     print("  Number of finished trials: ", len(study.trials))
#     print("  Number of pruned trials: ", len(pruned_trials))
#     print("  Number of complete trials: ", len(complete_trials))
#
#     print("Best trial:")
#     trial = study.best_trial
#
#     print("  Value: ", trial.value)
#
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))





model = define_model().to(DEVICE)

lr = 0.005
optimizer = getattr(optim, "RMSprop")(model.parameters(), lr=lr)
mse = nn.MSELoss()

# train model
loss_list = []
val_loss_list = []
iteration_number = 800

for iteration in range(iteration_number):
    # optimization
    optimizer.zero_grad()

    # Forward to get output
    results = model.forward(inputs)

    # Calculate Loss
    loss = mse(results.squeeze(), targets)

    # backward propagation
    loss.backward()

    # Updating parameters
    optimizer.step()

    # store loss
    loss_list.append(loss.data)

    val_results = model.forward(np_val_x)
    val_loss = mse(val_results.squeeze(), np_val_y).data
    val_loss_list.append(val_loss)

test_results = model.forward(np_test_x).squeeze()

with open('predictions.csv', "wt") as fp:
    writer = csv.writer(fp, delimiter=",")
    writer.writerow(["id", "cases"])

    for count, result in enumerate(test_results):
        writer.writerow([str(count), result.item()])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
axes[0].plot(range(iteration_number), loss_list)
axes[1].plot(range(iteration_number), val_loss_list)
fig.tight_layout()
plt.show()
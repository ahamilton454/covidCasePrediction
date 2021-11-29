from load_data import load_data
from helpers import convert_to_tensor, define_model, write_predictions, deep_interpolate_data
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
import enum
import csv



class Model(enum.Enum):
    OptunaNN = 1
    NN = 2
    RidgeRegression = 3
    Visualize = 4

DEVICE = torch.device("cpu")
FEATURE_COLMNS = ["Region", "Population", "Area", "Pop. Density", "Coastline", "Net migration",
                                    "Infant mortality", "GDP", "Literacy", "Phones", "Arable", "Crops", "Other",
                                    "Birthrate", "Deathrate", "Agriculture", "Industry", "Service",
                                    "Handwashing Facilities", "Extreme poverty", "Median age", "Life expectancy",
                                    "Human development index"]
TUNING = Model.NN

x_train, y_train, x_val, y_val, x_test = load_data()


def objective(trial):

    num_cols_drop = trial.suggest_int("num dropped columns", 0, 3)

    ftcols = FEATURE_COLMNS
    drops = []
    for num in range(0, num_cols_drop):
        suggestion = trial.suggest_categorical("drop {}".format(num), ftcols)
        # ftcols.remove(suggestion)
        drops.append(suggestion)


    # Train Set
    inputs = convert_to_tensor(x_train, drop=drops)
    targets = torch.from_numpy(y_train)

    # Val Set
    torch_val_x = convert_to_tensor(x_val, drop=drops)
    torch_val_y = torch.from_numpy(y_val)


    model = define_model(num_features=inputs.shape[1]).to(DEVICE)

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

        val_results = model.forward(torch_val_x)
        val_loss = mse(val_results.squeeze(), torch_val_y).data
        val_loss_list.append(val_loss)

        trial.report(val_loss, iteration_number)

    return val_loss_list[-1]




if TUNING == Model.OptunaNN:
    if __name__ == "__main__":
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50, timeout=1000)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
elif TUNING == Model.NN:

    drops = []
    # Train Set
    inputs = convert_to_tensor(x_train, drop=drops)
    targets = torch.from_numpy(y_train)

    # Val Set
    torch_val_x = convert_to_tensor(x_val, drop=drops)
    torch_val_y = torch.from_numpy(y_val)

    # Test Set
    np_test_x = convert_to_tensor(x_test, drop=drops)

    model = define_model(inputs.shape[1]).to(DEVICE)

    lr = 0.003
    optimizer = getattr(optim, "Adam")(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # train model
    loss_list = []
    val_loss_list = []
    iteration_number = 600

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

        val_results = model.forward(torch_val_x)
        val_loss = mse(val_results.squeeze(), torch_val_y).data
        val_loss_list.append(val_loss)

    test_results = model.forward(np_test_x).squeeze()
    print("Training Loss: {}".format(loss_list[-1]))
    print("Validation Loss: {}".format(val_loss_list[-1]))

    write_predictions(test_results)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    axes[0].plot(range(iteration_number), loss_list)
    axes[1].plot(range(iteration_number), val_loss_list)
    fig.tight_layout()
    plt.show()

elif TUNING == Model.RidgeRegression:
    from ridge_regression import RidgeModel
    from sklearn.metrics import mean_squared_error

    x = deep_interpolate_data(x_train)
    # Train Set
    inputs = convert_to_tensor(deep_interpolate_data(x_train), file_name="x_train.csv")
    targets = torch.from_numpy(y_train)

    # Val Set
    torch_val_x = convert_to_tensor(deep_interpolate_data(x_val), file_name="x_val.csv")
    torch_val_y = torch.from_numpy(y_val)

    # Test Set
    np_test_x = convert_to_tensor(x_test, file_name="x_test.csv")

    ridge = RidgeModel(alpha=1)

    ridge.fit(inputs, targets)

    print("Ridge Regression Training Loss: ", mean_squared_error(ridge.predict(inputs), targets))
    print("Ridge Regression Validation Loss: ", mean_squared_error(ridge.predict(torch_val_x), torch_val_y))
elif TUNING == Model.Visualize:

    def remove_none(arr):
        for i in range(0, len(arr)):
            for j in range(0, len(arr[0])):
                if arr[i, j] == None:
                    arr[i, j] = 0

        return arr

    arr_train = remove_none(np.array(x_train))
    arr_val = remove_none(np.array(x_val))

    fig, axes = plt.subplots(nrows=4, ncols=7)
    fig.tight_layout()
    for index, i in enumerate(FEATURE_COLMNS):
        plt.subplot(4, 7, index+1)
        plt.hist(arr_train[:, index], density=True, bins=10)  # density=False would make counts
        plt.hist(arr_val[:, index], density=True, bins=10)  # density=False would make counts
        plt.xlabel(i)

    plt.show()




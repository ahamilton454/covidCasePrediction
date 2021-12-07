import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
from ridge_regression import RidgeModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

FEATURE_COLMNS = ["Region", "Population", "Area", "Pop. Density", "Coastline", "Net migration",
                                    "Infant mortality", "GDP", "Literacy", "Phones", "Arable", "Crops", "Other",
                                    "Birthrate", "Deathrate", "Agriculture", "Industry", "Service",
                                    "Handwashing Facilities", "Extreme poverty", "Median age", "Life expectancy",
                                    "Human development index"]

def normalize_by_column(array):
    x = np.array(array, dtype=np.float32)
    norm_array = (x - x.min(0)) / x.ptp(0)
    return norm_array

def normalize_df(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df


def convert_to_tensor(arr, drop=[], file_name=""):
    df = pd.DataFrame(arr, columns=FEATURE_COLMNS)
    # df.to_csv(file_name)
    df = df.drop(drop, axis=1).fillna(df.mean())

    normalized_np = normalize_by_column(df.values)
    # numpy.savetxt(file_name, normalized_np)
    tensor = torch.from_numpy(normalized_np)
    return tensor

def convert_to_tensor_pca(arr, drop=[], file_name=""):

    from sklearn.decomposition import PCA

    df = pd.DataFrame(arr, columns=FEATURE_COLMNS)
    df = df.drop(drop, axis=1).fillna(df.mean())

    df = pd.DataFrame(arr, columns=FEATURE_COLMNS)
    df = (df - df.min()) / (df.max() - df.min())
    df = df.fillna(-1)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(df.values)

    normalized_np = normalize_by_column(principalComponents)
    tensor = torch.from_numpy(normalized_np)
    return tensor


def define_model(num_features=23):
    layers = [nn.Linear(num_features, 17),
              nn.Sigmoid(),
              nn.Dropout(0.25),
              nn.Linear(17, 7),
              nn.Softmax(),
              nn.Dropout(0.22),
              nn.Linear(7, 1)]
    return nn.Sequential(*layers)

def define_model_optuna(trial, num_features=23):

    layers = []

    n_layers = trial.suggest_int("n_layers", 1, 3)
    in_features = num_features
    for i in range(0, n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 32)
        layers.append(nn.Linear(in_features, out_features))
        layers.append([nn.ReLU(), nn.Softmax(), nn.Sigmoid()][trial.suggest_int("activation_l{}".format(i), 0, 2)])
        p = trial.suggest_float("dropout_l{}".format(i), 0, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features

    layers.append(nn.Linear(in_features, 1))


    return nn.Sequential(*layers)

def write_predictions(test_results):
    with open('predictions.csv', "wt") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(["id", "cases"])

        for count, result in enumerate(test_results):
            writer.writerow([str(count), result.item()])

def deep_interpolate_data(arr):
    df = pd.DataFrame(arr, columns=FEATURE_COLMNS)
    df = normalize_df(df)

    models = []

    for (columnName, columnData) in df.iteritems():

        # Copy df
        dfc = df.copy()

        # Remove drop rows with None in the working column
        dfc = dfc.dropna()

        # Set working column values as y
        y = dfc[[columnName]].to_numpy()

        # Set other columns as X
        X = dfc.drop([columnName], axis=1).fillna(dfc.mean()).to_numpy()

        model = DecisionTreeRegressor(max_depth=5)
        model.fit(X, y)
        models.append((columnName, model))


    def get_model(name):
        model = [item for item in models if item[0] == name][0][1]
        return model

    for (columnName, columnData) in df.iteritems():
        dfNeg = df.copy().drop([columnName], axis=1).fillna(df.mean())
        model = get_model(columnName)
        df[columnName] = dfNeg.apply(lambda row: model.predict(np.expand_dims(row.to_numpy(), axis=0)), axis=1)



    return df
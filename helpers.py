import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
from ridge_regression import RidgeModel
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


def define_model(num_features=23):
    layers = [nn.Linear(num_features, num_features), nn.ReLU(), nn.Linear(num_features, 1)]
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

    ridge_models = []

    for (columnName, columnData) in df.iteritems():

        # Copy df
        dfc = df.copy()

        # Remove drop rows with None in the working column
        dfc = dfc.dropna()

        # Set working column values as y
        y = dfc[[columnName]].to_numpy()

        # Set other columns as X
        X = dfc.drop([columnName], axis=1).fillna(dfc.mean()).to_numpy()

        model = RidgeModel(alpha=0.5)
        model.fit(X, y)
        ridge_models.append((columnName, model))


    def get_model(name):
        model = [item for item in ridge_models if item[0] == name][0][1]
        return model

    for (columnName, columnData) in df.iteritems():
        dfNeg = df.copy().drop([columnName], axis=1).fillna(df.mean())
        model = get_model(columnName)
        df[columnName] = dfNeg.apply(lambda row: model.predict(np.expand_dims(row.to_numpy(), axis=0)), axis=1)



    return df
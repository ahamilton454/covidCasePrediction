import numpy as np
import pandas as pd
import torch
import torch.nn as nn

FEATURE_COLMNS = ["Region", "Population", "Area", "Pop. Density", "Coastline", "Net migration",
                                    "Infant mortality", "GDP", "Literacy", "Phones", "Arable", "Crops", "Other",
                                    "Birthrate", "Deathrate", "Agriculture", "Industry", "Service",
                                    "Handwashing Facilities", "Extreme poverty", "Median age", "Life expectancy",
                                    "Human development index"]

def normalize_by_column(array):
    x = np.array(array, dtype=np.float32)
    norm_array = (x - x.min(0)) / x.ptp(0)
    return norm_array


def convert_to_tensor(arr, drop=[]):
    df = pd.DataFrame(arr, columns=FEATURE_COLMNS)
    df = df.drop(drop, axis=1).fillna(0)
    normalized_df = normalize_by_column(df.values)
    tensor = torch.from_numpy(normalized_df)
    return tensor


def define_model(num_features=23):
    layers = [nn.Linear(num_features, num_features), nn.ReLU(), nn.Dropout(0.3), nn.ReLU(), nn.Linear(num_features, 1), nn.ReLU()]
    return nn.Sequential(*layers)
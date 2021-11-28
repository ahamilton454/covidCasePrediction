import pandas as pd 
#save the prediction array test_pred into a valid submission file
pd.DataFrame(test_pred).to_csv("predictions.csv", header=["cases"], index_label="id")

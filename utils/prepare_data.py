import os

import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv(os.path.join("data", "raw", "train.csv"))

data_train, data_test = train_test_split(df, test_size=0.2, random_state=42)

data_train.to_csv(os.path.join("data", "processed", "train.csv"))
data_test.to_csv(os.path.join("data", "processed", "test.csv"))
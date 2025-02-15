import pandas as pd
import numpy as np
from pygam import LinearGAM, s


data_train = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data_test = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")


data_train['Timestamp'] = pd.to_datetime(data_train['Timestamp'])
data_train['year'] = data_train['Timestamp'].dt.year
data_train['month'] = data_train['Timestamp'].dt.month
data_train['day'] = data_train['Timestamp'].dt.weekday
data_train['hour'] = data_train['Timestamp'].dt.hour


data_test['Timestamp'] = pd.to_datetime(data_test['Timestamp'])
data_test['year'] = data_test['Timestamp'].dt.year
data_test['month'] = data_test['Timestamp'].dt.month
data_test['day'] = data_test['Timestamp'].dt.weekday
data_test['hour'] = data_test['Timestamp'].dt.hour

x_train = data_train[['month','day','hour']].values
y_train = data_train['trips'].values


model = LinearGAM(s(0) + s(1) + s(2))
modelFit = model.fit(x_train, y_train)


X_test = data_test[['month','day','hour']].values
pred = modelFit.predict(X_test)


predictions_df = pd.DataFrame({"hour_index": np.arange(len(y_pred)), "predicted_trips": y_pred})
predictions_df.to_csv("predictions.csv", index=False)
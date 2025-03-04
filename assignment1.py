import pandas as pd
import numpy as np
from pygam import LinearGAM, s


data_train = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data_test = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")



x_train = data_train[['month','day','hour']].values
y_train = data_train['trips'].values


model = LinearGAM(s(0) + s(1) + s(2))
modelFit = model.fit(x_train, y_train)


X_test = data_test[['month','day','hour']].values
pred = modelFit.predict(X_test)


predictions_df = pd.DataFrame({"hour_index": np.arange(len(pred)), "predicted_trips": pred})
predictions_df.to_csv("predictions.csv", index=False)
from pygam import LinearGAM, s, f, l
import pandas as pd

data_train = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data_test = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")

x_train = data_train[['month','day','hour']]
y_train = data_train['trips']

model = LinearGAM(s(0) + s(1) + s(2))
modelFit = model.gridsearch(x_train.values, y_train)

X_data_test = data_test[['month','day','hour']]
y_pred = modelFit.predict(X_data_test)


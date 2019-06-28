import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# Intro to ML

flights = pd.read_csv('data/flights.csv')
flights = flights[['dep_delay', 'month', 'day', 'sched_dep_time', 'sched_arr_time', 'carrier', 'origin', 'distance']]
flights = flights.dropna()

# EDA
flights['dep_delay'].describe()
target_dist = sns.distplot(flights['dep_delay'])
plt.show()

# dummy encode
flights_matrix = pd.get_dummies(flights)
# pd.get_dummies(flights, drop_first = True)
# pd.get_dummies(flights, dummy_na = True)

# data partitioning
from sklearn import model_selection
flights_train, flights_test = model_selection.train_test_split(flights_matrix, test_size = 0.2, random_state = 123)

# get features and target
X = flights_train.drop(columns = 'dep_delay')
Y = flights_train['dep_delay']
X_test = flights_test.drop(columns = 'dep_delay')
Y_test = flights_test['dep_delay']

# apply linear model
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()   # instantiate model
linear_model.fit(X, Y)              # train model

# model validation
from sklearn.metrics import mean_squared_error

## training data
mse = mean_squared_error(Y, linear_model.predict(X))
np.sqrt(mse)

## testing data
np.sqrt(mean_squared_error(Y_test, linear_model.predict(X_test)))

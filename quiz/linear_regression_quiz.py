import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Assign the dataframe to this variable.
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
X = np.reshape(bmi_life_data['BMI'].as_matrix(), (-1, 1))
y = np.reshape(bmi_life_data['Life expectancy'].as_matrix(), (-1, 1))

# Make and fit the linear regression model
bmi_life_model = LinearRegression()
bmi_life_model.fit(X, y)

# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict([21.07931])

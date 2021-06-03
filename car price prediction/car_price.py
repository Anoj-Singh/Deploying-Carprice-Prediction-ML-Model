

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 


df = pd.read_csv("car data.csv")
df.head()

df.drop('Car_Name', axis=1, inplace=True)
df.head()

current_year = 2021
df['Year'] = current_year - df['Year']
df.head()

ohe = LabelEncoder()
df['Fuel_Type'] = ohe.fit_transform(df['Fuel_Type']) 
df['Seller_Type'] = ohe.fit_transform(df['Seller_Type'])  
df['Transmission'] = ohe.fit_transform(df['Transmission'])  

x = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
x[:5], y[:5]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

from sklearn.ensemble import GradientBoostingRegressor
grad_model = GradientBoostingRegressor()

grad_model.fit(x_train, y_train)
print(grad_model.score(x_test, y_test))

import pickle
pickle.dump(grad_model, open('car_price_prediction_model.pkl', 'wb'))



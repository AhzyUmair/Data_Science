import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv=pd.read_csv('50_Startups.csv')
x=csv.iloc[:,:-1].values
y=csv.iloc[:,-1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer([("encoder",OneHotEncoder(),[3])],remainder="passthrough")
x=np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_tr,y_tr)
y_pred=lr.predict(x_ts)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_ts.reshape(len(y_ts),1)),1))
print(lr.predict([[0,0,1,123455,12344,78902]]))

print(lr.coef_)
print(lr.intercept_)

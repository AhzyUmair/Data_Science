import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv=pd.read_csv('Data.csv')
x=csv.iloc[:,:-1].values
y=csv.iloc[:,-1].values

from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy='median')
si.fit(x[:,1:3])
x[:,1:3]=si.transform(x[:,1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x=np.array(ct.fit_transform(x))

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtr=sc.fit_transform(xtr)
xts=sc.transform(xts)

from sklearn.linear_model import LinearRegression                #Extra Part for simple Linear Regression starts here.
lr=LinearRegression()
lr.fit(xtr,ytr)

yp=lr.predict(xts)

from sklearn.metrics import r2_score
print(r2_score(yts, yp))                                         ##Extra Part for simple Linear Regression ends here.


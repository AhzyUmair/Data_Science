import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cs=pd.read_csv('Data.csv')
x=cs.iloc[:,:-1].values
y=cs.iloc[:,-1].values

from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy="median")
si.fit(x[:,1:3])
x[:,1:3]=si.transform(x[:,1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],remainder='passthrough')
x=np.array(ct.fit_transform(x))

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y) 
from sklearn.model_selection import train_test_split
atr,ate,btr,bte=train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
atr=sc.fit_transform(atr)
ate=sc.transform(ate)

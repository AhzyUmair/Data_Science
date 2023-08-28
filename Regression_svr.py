import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv=pd.read_csv('Position_Salaries.csv')
x=csv.iloc[:,1:-1].values
y=csv.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
sc1=StandardScaler()
y=y.reshape(len(y),1)
y=sc1.fit_transform(y)

from sklearn.svm import SVR
svr=SVR(kernel='rbf')
svr.fit(x,y)

pred=sc1.inverse_transform(svr.predict(sc.transform([[6.5]])).reshape(-1,1))

xgrid=np.arange(min(x),max(x),0.1)
xgrid=xgrid.reshape(len(xgrid),1)

plt.scatter(sc.inverse_transform(x),sc1.inverse_transform(y),color='red')
plt.plot(sc.inverse_transform(xgrid),sc1.inverse_transform(svr.predict(xgrid).reshape(-1,1)),color='blue')
plt.show()

plt.scatter(sc.inverse_transform(x),sc1.inverse_transform(y),color='red')
plt.plot(sc.inverse_transform(x),sc1.inverse_transform(svr.predict(x).reshape(-1,1)),color='blue')
plt.show()

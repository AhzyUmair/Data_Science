import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv=pd.read_csv("Position_Salaries.csv")
x=csv.iloc[:,1:-1].values
y=csv.iloc[:,-1].values

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=0)
dt.fit(x,y)

print(dt.predict([[100]]))

x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)

plt.scatter(x,y,color='red')
plt.plot(x_grid,dt.predict(x_grid),color='black')
plt.show()

plt.scatter(x,y,color='red')
plt.plot(x,dt.predict(x),color='black')
plt.show()

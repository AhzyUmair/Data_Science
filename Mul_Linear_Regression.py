import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv=pd.read_csv('Position_Salaries.csv')
x=csv.iloc[:,1:-1].values
y=csv.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)
ypd=lr.predict(x)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
x_poly=poly.fit_transform(x)
poly.transform(x)
lin=LinearRegression()
lin.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,lr.predict(x),color='blue')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()

plt.scatter(x,y,color='red')
plt.plot(x,lin.predict(poly.fit_transform(x)),color='blue')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()

print(lr.predict([[7.5]]))
print(lin.predict(poly.fit_transform([[7.5]])))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
csv=pd.read_csv('Social_Network_Ads.csv')
x=csv.iloc[:,:-1].values
y=csv.iloc[:,-1].values
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(x,y,train_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtr=sc.fit_transform(xtr)
xts=sc.transform(xts)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(xtr,ytr)
ypr=lr.predict(xts)
a=np.concatenate((ypr.reshape(len(ypr),1),yts.reshape(len(yts),1)),1)
from sklearn.metrics import confusion_matrix,accuracy_score
cn=confusion_matrix(yts,ypr)
print(cn)
accuracy_score(yts,ypr)
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(xtr), ytr
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, lr.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(xts), yts
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, lr.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

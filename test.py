import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

boston = pd.read_csv('house.csv',header=0)
print(boston.head())

X = boston.drop('SalePrice',axis=1)
Y = boston['SalePrice']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

regressor = LinearRegression()

regressor.fit(x_train,y_train) 





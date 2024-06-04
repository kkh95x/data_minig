import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class UsdToTRY:
    def __init__(self):
        self.data = pd.read_csv('USD_TRY Historical 6-years.csv')
    
    def checkData(self):
        print(f'head {"-"*100}\n{self.data.head()}')
        print(f'describe{"-"*100}\n{self.data.describe()}')
        print(f'is Null {"-"*100}\n{self.data.isnull().sum()}')
    
    def dataPreprocessing(self):
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.drop_duplicates(subset=['Date'])
        self.data = self.data.sort_values(by='Date', ascending=False)
        self.data['day'] = self.data['Date'].dt.day
        self.data['month'] = self.data['Date'].dt.month
        self.data['year'] = self.data['Date'].dt.year
        self.data = self.data.drop('Vol.', axis=1)
        self.data.dropna(inplace=True)
        self.X = self.data[['day', 'month', 'year']]
        self.Y = self.data['Price']

    def showOutputDataPreprocessing(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.data['Date'], self.data['Price'])
        plt.title('Dollar Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()

    def buildAndTrainTheLinearModel(self):
        self.linear_model = LinearRegression()
        self.linear_model.fit(self.X, self.Y)
    
    def evaluateTheLinearModel(self):
        y_pred = self.linear_model.predict(self.X)
        mse = mean_squared_error(self.Y, y_pred)
        print(f'Mean Squared Error (Linear Regression): {mse}')
        plt.figure(figsize=(10, 5))
        plt.plot(self.data["Date"],y_pred, label='Predicted Price (Linear)')
        plt.plot(self.data["Date"],self.Y, label='Actual Price')
        plt.title('Actual vs Predicted Dollar Price (Linear)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def buildAndTrainTheDecisionTreeModel(self):
        self.decision_tree_model = DecisionTreeRegressor()
        self.decision_tree_model.fit(self.X, self.Y)
    
    def evaluateTheDecisionTreeModel(self):
        y_pred = self.decision_tree_model.predict(self.X)
        mse = mean_squared_error(self.Y, y_pred)
        print(f'Mean Squared Error (Decision Tree): {mse}')
        plt.figure(figsize=(10, 5))
        plt.plot(self.data["Date"],y_pred, label='Predicted Price (Decision Tree)',)
        plt.plot(self.data["Date"],self.Y,"--", label='Actual Price',)
        plt.title('Actual vs Predicted Dollar Price (Decision Tree)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def buildAndTrainTheKNNModel(self):
        self.knn_model = KNeighborsRegressor(n_neighbors=7) 
        self.knn_model.fit(self.X, self.Y)
    
    def evaluateTheKNNModel(self):
        y_pred = self.knn_model.predict(self.X)
        mse = mean_squared_error(self.Y, y_pred)
        print(f'Mean Squared Error (KNN): {mse}')
        plt.figure(figsize=(10, 5))
        plt.plot(self.data["Date"],y_pred, label='Predicted Price (KNN)')
        plt.plot(self.data["Date"],self.Y, label='Actual Price')
        plt.title('Actual vs Predicted Dollar Price (KNN)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    
    def predict(self, day, month, year, model_type='linear'):
        model = {
            'linear': self.linear_model,
            'decision_tree': self.decision_tree_model,
            'knn': self.knn_model
        }[model_type]
        y_pred = model.predict(pd.DataFrame({
            'day': [day],
            'month': [month],
            'year': [year]
        }))
        print(f'Predicted price on {day}-{month}-{year} with {model_type} model: {y_pred[0]}')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
class UsdToTRY:
    def __init__(self) :
        # self.data = pd.read_csv('USD_TRY Historical Data.csv')
        self.data = pd.read_csv('USD_TRY Historical 6-years.csv')

        pass
    def checkData(self):
        # Display the first few rows

        print(f'head {"-"*100}\n{self.data.head()}')

        # Basic statistics
        print(f'describe{"-"*100}\n{self.data.describe()}')

        # Check for missing values
        print(f'is Null {"-"*100}\n{self.data.isnull().sum()}')

    def dataPreprocessing(self):
        self.data['Date'] = pd.to_datetime(self.data['Date'])


        self.data = self.data.drop_duplicates(subset=['Date'])
        self.data = self.data.sort_values(by='Date',ascending=False)
        self.data['day']=self.data['Date'].dt.day
        self.data['month']=self.data['Date'].dt.month
        self.data['year']=self.data['Date'].dt.year

        # # Set 'Date' as the index
        # self.data.set_index('Date', inplace=True)

        #drop Empty Column
        self.data=self.data.drop('Vol.', axis=1)
        self.data.dropna(inplace=True)
        self.X=self.data[['day','month','year']]
        self.Y=self.data['Price']  

    def showOutputDataPreprocessing(self):
            plt.figure(figsize=(10,5))
            plt.plot(self.data['Date'],self.data['Price'])
            plt.title('Dollar Price Over Time')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.show()

    

    def buildAndTrainTheModel(self):
        # Initialize the model object
        self.model = LinearRegression()
        # Training the model
        self.model.fit(self.X,self.Y)
    
    def evaluateTheModel(self):
        # Make predictions
        # print(self.X_test)
        y_pred = self.model.predict(self.data[['day','month','year']])


        # Calculate the mean squared error
        mse = mean_squared_error(self.Y, y_pred)
        print(f'Mean Squared Error: {mse}')

        # Plot actual vs predicted prices
        plt.figure(figsize=(10,5))
        plt.plot(y_pred, label='Predicted Price')
        plt.plot(self.Y, label='Actual Price')
        plt.title('Actual vs Predicted Dollar Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    def predict(self,day,month,year):
   

        
        y_pred = self.model.predict(pd.DataFrame({
             "day":day,
             "month":month,
             "year":year
        },index=[0]))
        print(f'Predicted price on {day}-{month}-{year}---> {y_pred[0]}')


    def predictMonth(self):
       y_pred= self.model.predict(pd.DataFrame({
            "day":range(1,30),
            "month":[10 for x in range(1,30)],
            "year":[2024 for x in range(1,30)],

        }))
       
       print(y_pred)

         
              
         



plt.show()



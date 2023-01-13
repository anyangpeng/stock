import pandas as pd
import numpy as np
# import mlflow as mf
# from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from pulldata import YahooFinanceHistory
import time

class Model():
    def __init__(self,dataframe,features,target,lag=2, ratio=0.8,epochs=100, optimizer = 'Adam'):
        self.data = dataframe
        self.size = len(self.data)
        self.lag = lag
        self.splitratio = ratio
        self.features = features
        self.target = target
        self.epochs = epochs
        self.optimizer = optimizer

    def checkdata(self):
        """
        Performs ADF test to check if the time series data is stationary
        null hypothesis: non-stationary
        alternative:    stationary
        if p-vale < critical value then data is stationary
        """
        print(f'\n Performing ADF-test on {self.target}...')
        test = adfuller(self.data[self.target])
        if test[1] < 0.05:
            print('\n The data is stationary')
        else:
            print('\n The data is non-stationary')

        time.sleep(1)

    def splitdata(self):
        """
        Split the data into train/test sets:
            older data are used to train the model,
            newer data are used to test the model

        Scale data

        """
        # Taking care of datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['Date'] = self.data['Date'].map(lambda v: int(v.value))
        self.data = self.data.dropna()
        print('\n Splitting data...')

        trainSize = int(self.size * self.splitratio)
        self.trainData, self.testData = self.data[:trainSize], self.data[trainSize:]
        
        print('\n Number of Train and Test data: ', len(self.trainData), len(self.testData))

    def _split(self, X, y):
        """
        X-train is a 3-D vector: 
            for each x-train, 3-previous date are used (lag = 3)
            for each date, Open, Close, High Low Volume are used
        """
        tempX, tempy = [], []
        for i in range(len(X)-self.lag):
            tempX.append(X[i:i+self.lag])
            tempy.append(y[i+self.lag])
        return np.array(tempX), np.array(tempy)

    def processdata(self):

        scaler=StandardScaler()
        trainLabel = np.array(self.trainData[self.target])
        trainData = np.array(scaler.fit_transform(self.trainData[self.features]))
        testLabel = np.array(self.testData[self.target])
        testData = np.array(scaler.transform(self.testData[self.features]))
        
        self.X_train, self.y_train = self._split(trainData,trainLabel)
        self.X_test, self.y_test = self._split(testData,testLabel)

        print('\n Data ready to use.')
                
    def train(self):
        """
        Initialize and train model
        """
        # mf.tensorflow.autolog()
        self.model = Sequential()
        self.model.add(LSTM(32, activation='relu', return_sequences=False, input_shape=(self.X_train.shape[1:])))
        self.model.add(Dense(1))
        self.model.compile(optimizer=self.optimizer, loss='mse')
        print('\n Training model...')
        self.history = self.model.fit(self.X_train, self.y_train, epochs = self.epochs, batch_size=64, verbose=0, validation_split= 0.1)
        print('\n Training finished!')

    def evaluate(self):
        """
        Creat training plots.
        """
        
        fig, axes = plt.subplots(1,2)
        axes[0].plot(self.history.history['loss'],'r')
        axes[0].plot(self.history.history['val_loss'],'b')
        axes[0].set_xlabel('epochs')
        axes[0].set_ylabel('MSE loss')
        axes[0].legend(['loss','val-loss'])
        axes[1].plot(self.y_test,'r.-')
        axes[1].plot(self.model.predict(self.X_test),'b.-')
        axes[1].set_xlabel('Recent Days')
        axes[1].set_ylabel(f'{self.target}')
        axes[1].set_xticks([])
        axes[1].legend(['Actual','Prediction'])
        fig.tight_layout()
        fig.savefig('result.jpg')
        # return fig

    def tomorrow(self):
        '''
        Generate prediction for tomorrow
        '''

        return self.model.predict(np.array(self.X_test[-self.lag:]))[0][0], self.y_test[-1]

if __name__ == '__main__':
    model = Model(YahooFinanceHistory('AAPL', days_back=1800).get_quote(),\
        features = ['Open','Close','High','Low','Volume'], target = 'Adj Close')
    # model.checkdata()
    model.splitdata()
    model.processdata()
    model.train()
    model.evaluate()


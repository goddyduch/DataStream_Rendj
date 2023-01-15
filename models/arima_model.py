from statsmodels.tsa.arima.model import ARIMA
from river import time_series
from river import linear_model
from river import preprocessing
import numpy as np
import warnings

class  ARIMA_WINDOW():
    def __init__(self, p, d, q, window_size):

        self.p = p
        self.d = d
        self.q = q
        self.history = []
        self.window_size = window_size
        self.model_trained = False
        self.fitted_model = None

    def learn_one(self, y, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.history.append(x)
            if len(self.history) == self.window_size:
                self.model_trained = True
                model = ARIMA(np.array(self.history),
                                            order=(self.p,self.d,self.q))
                model.initialize_approximate_diffuse()
                self.fitted_model = model.fit()
                
            elif len(self.history) > self.window_size:
                self.history.pop(0)
                model = ARIMA(np.array(self.history),
                                            order=(self.p,self.d,self.q))
                model.initialize_approximate_diffuse()
                self.fitted_model = model.fit()
            
    def predict_one(self,x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.model_trained:
                return self.fitted_model.forecast()[0]
            else:
                raise Exception

class ONLINE_ARIMA():
    def __init__(self, p, d, q):

        self.history = []
        self.model = time_series.SNARIMAX(p,d,q,regressor=( preprocessing.StandardScaler() | linear_model.LinearRegression() ))

    def learn_one(self, y, x):
        self.model = self.model.learn_one(x)
            
    def predict_one(self,x):
        return self.model.forecast(horizon=1)[0]


from models.arima_model import ARIMA_WINDOW
from models.arima_model import ONLINE_ARIMA
from models.linear_regression_model import ONLINE_REGRESSION
from models.linear_regression_model import BATCH_REGRESSION
from river import linear_model
from river.utils import dict2numpy
from river.stream import iter_pandas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from river import preprocessing
import yfinance as yf


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


class  Online_trainer():
    def __init__(self, key_to_predict, window_size, arima_args=(5,1,1), online_args=(5,1,1), online_reg_args=(None,None, 0.,0.001), batch_reg_args=(0.001,)):

        arima_args_ = arima_args + (window_size,)
        batch_reg_args_ = (window_size,) + batch_reg_args
        self.arima_model = ARIMA_WINDOW(*arima_args_)
        self.online_arima_model = ONLINE_ARIMA(*online_args)
        self.online_regressor = ONLINE_REGRESSION(*online_reg_args)
        self.batch_regressor = BATCH_REGRESSION(*batch_reg_args_)
        self.window_size = window_size
        self.reach_window = False
        self.n_iterate = 0
        self.previous_x = None
        # self.scaler = preprocessing.StandardScaler()
        self.key_to_predict = key_to_predict


    def learn_one(self, x_y):
        self.n_iterate += 1
        if self.n_iterate == self.window_size:
            self.reach_window = True
        
        input = x_y
        label = x_y[self.key_to_predict]

        ###online_regressor
        self.online_regressor.learn_one(input,label)
        self.batch_regressor.learn_one(input,label)
        ####arima
        self.arima_model.learn_one(input,label)
        self.online_arima_model.learn_one(input,label)

            
    def predict_one(self, x_y):
        if self.reach_window:
            input = x_y
            label = x_y[self.key_to_predict]

            result_online_reg = self.online_regressor.predict_one(input)
            result_bacth_reg = self.batch_regressor.predict_one(input)
            result_arima_reg = self.arima_model.predict_one(input)
            result_online_arima = self.online_arima_model.predict_one(input)

            return label, result_online_reg, result_bacth_reg,result_arima_reg, result_online_arima
        else:
            raise ValueError


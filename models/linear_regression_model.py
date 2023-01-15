from river import linear_model
from river import preprocessing
from river.utils import dict2numpy
from sklearn.linear_model import Lasso, LinearRegression
import numpy as np
import warnings


class  ONLINE_REGRESSION():
    def __init__(self, optimizer, loss, l2, l1):

        self.model = (
            preprocessing.AdaptiveStandardScaler() |
            linear_model.LinearRegression(optimizer, loss, l2, l1)
        )
        self.first_iterate = True
        self.past_x = None
        self.past_predict = 0

    def learn_one(self, x_future, y_to_pred):
        if self.first_iterate:
            self.first_iterate = False
        else:
            self.model.learn_one(self.past_x,y_to_pred)
        self.past_x = x_future

            
    def predict_one(self,x):
        return self.model.predict_one(x)

class BATCH_REGRESSION():

    def __init__(self, window_size=100, lbda=0.):
        self.h = None
        # TODO

        #####Initialization of batch

        self.X_batch = []
        self.y_batch = []
        self.lbda = lbda
        
        self.window_size = window_size

    def learn_one(self, x, y=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(self.X_batch) == self.window_size:
                self.X_batch.pop(0)
                self.y_batch.pop(0)
            
            
            if len(self.X_batch) >= 1:
                self.y_batch.append(y)
                if self.lbda > 0.:
                    self.h = Lasso(alpha=self.lbda)
                else:
                    self.h = LinearRegression()
                self.h.fit(self.X_batch, self.y_batch)

            self.X_batch.append(dict2numpy(x))

            return self

    def predict_one(self, x):
        if len(self.X_batch) == 0:
            return 0
        
        else:
            return self.h.predict( np.array([dict2numpy(x)]) )[0]
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import argparse
from river.stream import iter_pandas
from online_training import Online_trainer
from get_features import get_features_for_past_data
import numpy as np
from sklearn.metrics import r2_score

def trunc(a,x):
    temp = str(a)
    for i in range(len(temp)):
        if temp[i] == '.':
            try:
                return temp[:i+x+1]
            except:
                return temp

def pair(arg):
    return [str(x) for x in arg.split(',')]

def training_loop_simulate_stream(stream, trainer, window_model, window_metric):

    y_true = []
    y_pred_online_reg = []
    y_pred_batch_reg = []
    y_pred_arima = []
    y_pred_online_arima = []
    Result = ""
    for i, x_ in enumerate(stream):
        x_y = x_[0]
        if i >= window_model:
            label, y_or, y_br,y_ar, y_oar = trainer.predict_one(x_y)
            print("GT: ", trunc(label, 6), Result)
            Result = "Predicted of ARIMA :" + trunc(y_ar,6) +" ,O ARIMA :"+ trunc(y_oar, 6) + " ,O REG :"+ trunc(y_or, 6) +  " ,BATCH REG :"+ trunc(y_br, 6)
            y_true.append(label)
            y_pred_online_reg.append(y_or)
            y_pred_batch_reg.append(y_br)
            y_pred_arima.append(y_ar)
            y_pred_online_arima.append(y_oar)
        trainer.learn_one(x_y)
    

    y_true = y_true[1:]
    del y_pred_online_reg[-1]
    del y_pred_batch_reg[-1]
    del y_pred_arima[-1]
    del y_pred_online_arima[-1]

    y_true = np.array(y_true)
    y_pred_online_reg = np.array(y_pred_online_reg)
    y_pred_batch_reg = np.array(y_pred_batch_reg)
    y_pred_arima = np.array(y_pred_arima)
    y_pred_online_arima = np.array(y_pred_online_arima)

    R2_online_reg = []
    R2_batch_reg = []
    R2_arima = []
    R2_online_arima = []


    for i in range(window_metric,len(y_true)):
        if i-window_model >= 0:
            y_true_ = y_true[i-window_model:i]
            y_pred_online_reg_ = y_pred_online_reg[i-window_model:i]
            y_pred_batch_reg_ = y_pred_batch_reg[i-window_model:i]
            y_pred_arima_ = y_pred_arima[i-window_model:i]
            y_pred_online_arima_ = y_pred_online_arima[i-window_model:i]
        else:
            y_true_ = y_true[:i]
            y_pred_online_reg_ = y_pred_online_reg[:i]
            y_pred_batch_reg_ = y_pred_batch_reg[:i]
            y_pred_arima_ = y_pred_arima[:i]
            y_pred_online_arima_ = y_pred_online_arima[:i]
        R2_online_reg.append(r2_score(y_true_,y_pred_online_reg_))
        R2_batch_reg.append(r2_score(y_true_,y_pred_batch_reg_))
        R2_arima.append(r2_score(y_true_,y_pred_arima_))
        R2_online_arima.append(r2_score(y_true_,y_pred_online_arima_))


    plt.plot(R2_online_reg, label="online regression")
    plt.plot(R2_batch_reg, label="batch regression")
    plt.plot(R2_arima,  label="window arima")
    plt.plot(R2_online_arima, label="online arima")
    plt.legend()
    plt.show()

    plt.plot(y_true,  label="real data")
    plt.plot(y_pred_online_reg, label="online regression")
    plt.plot(y_pred_batch_reg, label="batch regression")
    plt.plot(y_pred_arima, label="window arima")
    plt.plot(y_pred_online_arima, label="online arima")
    plt.axis([0, len(y_true), min(y_true)*0.8, max(y_true)*1.2])
    plt.legend()
    plt.show()



def main(args):

    ticker_list = args.ticker_list
    refresh_time = args.refresh_time
    time_serie_size = args.time_serie_size
    window_size = args.window_size
    key_to_predict = args.key_to_predict
    window_metric = args.window_metric

    data = {}
    result = {}
    for market_name, index in ticker_list:
        data_ = yf.Ticker(index)
        data[market_name] = data_.history(period=refresh_time, start=time_serie_size[0], end=time_serie_size[1]).copy()
    

    data = get_features_for_past_data(data, "Close", window_size)
    for market_name, _ in ticker_list:
        trainer = Online_trainer(key_to_predict=key_to_predict,window_size = window_size )
        result[market_name] = training_loop_simulate_stream(iter_pandas(data[market_name]), trainer, window_size, window_metric)


    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh_time', help='refresh time in day',
                        default='1d', type=str)
    parser.add_argument('-ticker_list',type=pair, nargs='+', 
                        default=[("cac40","^FCHI"), ("Nasdaq","NQ=F"), ('Topix Index', 'TPX')])
    
    parser.add_argument('-time_serie_size',type=pair, nargs='+', 
                    default=['2010-1-1', '2022-12-19'])

    parser.add_argument('--window_size', help='window size for batch method',
                    default=100, type=int)
    
    parser.add_argument('--window_metric', help='window size for batch method',
                default=200, type=int)

    parser.add_argument('--key_to_predict', help='key to predict ex Open',
                        default='Returns', type=str)
        

    main(parser.parse_args())
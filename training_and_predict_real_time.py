from get_features import Feature_real_time
from kafka import KafkaConsumer
import json
import argparse
from online_training import Online_trainer

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

def training_loop_real_stream(input, trainer, window_model, i, name):
    if i >= window_model:
        label, y_or, y_br, y_ar, y_oar = trainer.predict_one(input)
        print("Current return of {}:".format(name),trunc(label,6),"Prediction with ARIMA :", trunc(y_ar,6) ,", O ARIMA :", trunc(y_oar,6), ", O REGRESSION :", trunc(y_or,6), "B REGRESSION :", trunc(y_br,6))
        print("")
    trainer.learn_one(input)


def main(args):

    topic_name = args.name_topic
    key_to_predict = args.key_to_predict
    window_size = args.window_size

    consumer = KafkaConsumer(topic_name, bootstrap_servers="localhost:9092",value_deserializer=lambda x: json.loads(x.decode('utf-8')))
    tickers = []
    trainers = {}
    for i in range(len(args.ticker_list)):
        tickers.append(args.ticker_list[i][0])
        trainers[args.ticker_list[i][0]] = Online_trainer(key_to_predict=key_to_predict,window_size = window_size)

    i = 0
    feature_exctractor = Feature_real_time(tickers, useful_index="Close", window_size=window_size)
    for msg in consumer:
        msg = json.loads(msg.value)
        stream_data_with_features = feature_exctractor.get_feature_real_time_stream(msg)
        for key in stream_data_with_features.keys():
            training_loop_real_stream(stream_data_with_features[key], trainers[key], window_size, i, key)
        i+=1
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_topic', default="real_time_stock", type=str)
    parser.add_argument('-ticker_list',type=pair, nargs='+', 
                        default=[("cac40","^FCHI"), ("Nasdaq","NQ=F"), ('Topix Index', 'TPX')])
    parser.add_argument('--key_to_predict', help='key to predict ex Open',
                    default='Returns', type=str)
    parser.add_argument('--window_size', help='window size for batch method',
                default=10, type=int)

    main(parser.parse_args())
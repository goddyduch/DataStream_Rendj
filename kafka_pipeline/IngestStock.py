from kafka import KafkaProducer

import pandas_datareader as web
import pandas as pd 
import json
import time
from datetime import datetime
import argparse

#OUVRIR KAFKA 
# bin/zookeeper-server-start.sh config/zookeeper.properties
# bin/kafka-server-start.sh config/server.properties


#A METTRE EN LIGNE DE COMMANDE  CREATION TOPIC 
#bin/kafka-topics.sh --create --topic refresh_time --bootstrap-server localhost:9092

def pair(arg):
    return [str(x) for x in arg.split(',')]

def main(args): 

    refresh_time = args.refresh_time
    name_topic = args.name_topic
    ticker = args.ticker_list

    producer = KafkaProducer(bootstrap_servers="localhost:9092",value_serializer=lambda v: json.dumps(v).encode('utf-8'))


    while True:
        timeout = time.time() + 60*refresh_time

        stocks = {}
        max_stocks = {}
        min_stocks = {}
        for ticker_name, indice in ticker:
            stocks[ticker_name] = []
            max_stocks[ticker_name] = 0
            min_stocks[ticker_name] = 1e99

        while True : 
            for ticker_name, indice in ticker:
                current_price = web.get_quote_yahoo(indice)["regularMarketPrice"][indice]
                stocks[ticker_name].append(current_price)

                if max_stocks[ticker_name] < current_price:
                    max_stocks[ticker_name] = current_price
                if min_stocks[ticker_name] > current_price:
                    min_stocks[ticker_name] = current_price
            
            if time.time() > timeout:
                break
        
        stock_interval = {}
        current_dateTime = datetime.now()
        for ticker_name, indice in ticker:
            stock_interval[ticker_name] = {'Date' : current_dateTime,
                                      'Open' : stocks[ticker_name][0],
                                      'High' : max_stocks[ticker_name],
                                      'Low'  : min_stocks[ticker_name],
                                      'Close': stocks[ticker_name][-1]}
        
        message = json.dumps(stock_interval, default=str)
        producer.send(name_topic, message)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Generating temperature models')

    parser.add_argument('--refresh_time', help='refresh time in minute',
                        default=1/60, type=int)
    parser.add_argument('-ticker_list',type=pair, nargs='+', 
                        default=[("cac40","^FCHI"), ("Nasdaq","NQ=F"), ('Topix Index', 'TPX')])
    parser.add_argument('--name_topic', default="real_time_stock", type=str)

    main(parser.parse_args())
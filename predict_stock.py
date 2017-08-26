import tensorflow as tf
from pandas_datareader import data
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from utils import scale_individual_stock

symbol = 'BAC'

past_month = datetime.now() - timedelta(days=30)

try:
    data = data.DataReader(symbol.upper(), 'google', past_month.strftime("%Y-%m-%d"))
except:
    raise ValueError('Symbol not found')

variables_dict = {}

SY = np.log(data['Close']) - np.log(data['Close'].shift())
positive = pd.Series(np.where(data['Close'] >= data['Open'], 1, 0))
for days in [5, 10, 15, 20, 25]:
    variables_dict['MA' + str(days)] = data['Close'][days*(-1) - 1:-1].mean()
    variables_dict['BIAS' + str(days)] = (data['Close'][-1] - variables_dict['MA' + str(days)]) / variables_dict['MA' + str(days)]
    variables_dict['PSY' + str(days)] = positive[days*(-1):].sum()
for days in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]:
    variables_dict['ASY' + str(days)] = SY[days * (-1):].mean()

responses = {'daily': None, 'weekly': None, 'bi_weekly': None, 'monthly': None}

scale_individual_stock(variables_dict)

for response in responses:
    with open('saved_models/' + response + '/' + 'predictors.json', 'r') as fd:
        predictor_names = json.loads(fd.read())

    predictors = np.column_stack(([variables_dict[variable_name] for variable_name in predictor_names]))
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        saver = tf.train.import_meta_graph('saved_models/' + response + '/model.meta')
        saver.restore(sess, 'saved_models/' + response + '/model')
        x = graph.get_tensor_by_name("x:0")
        weights = graph.get_tensor_by_name("weights:0")
        biases = graph.get_tensor_by_name("biases:0")
        output = tf.add(tf.matmul(x, weights), biases)
        pred = tf.nn.softmax(output)
        responses[response] = sess.run(pred, feed_dict={x: predictors})

print("Projections for {}:".format(symbol))
for key, value in responses.items():
    print('{} probability of price going up: {}%'.format(key, round(value[0,1]*100, 2)))

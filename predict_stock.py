import tensorflow as tf
from pandas_datareader import data
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from utils import scale_individual_stock

r"""
This script uses locally saved models to make predictions about a stock's future price. First, a stock's past data is 
downloaded using Google Finance, next the stock's performance metrics are calculated, then the various prediction models 
are loading from file, finally the stock's metrics are run through the models to make predictions about the price and 
its direction over different time periods.
"""

# Download stock data from Google Finance
symbol = 'BAC'
past_month = datetime.now() - timedelta(days=30)
try:
    data = data.DataReader(symbol.upper(), 'google', past_month.strftime("%Y-%m-%d"))
except:
    raise ValueError('Symbol not found')

# Calculate performance metrics
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

# Load various classification models and run stock metrics through the models
for response in responses:
    with open('saved_models/' + response + '/' + 'predictors.json', 'r') as fd:
        predictor_names = json.loads(fd.read())
    predictors = np.column_stack(([variables_dict[variable_name] for variable_name in predictor_names]))
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        saver = tf.train.import_meta_graph('saved_models/' + response + '/Classification/model.meta')
        saver.restore(sess, 'saved_models/' + response + '/Classification/model')
        x = graph.get_tensor_by_name("x:0")
        weights = []
        biases = []
        for layer_num in range(100):
            try:
                weights.append(graph.get_tensor_by_name("weights_{}:0".format(layer_num)))
                biases.append(graph.get_tensor_by_name("biases_{}:0".format(layer_num)))
            except:
                num_layers = layer_num - 1
                break
        layer = x
        for layer_num in range(num_layers):
            layer = tf.nn.relu(tf.add(tf.matmul(layer, weights[layer_num]), biases[layer_num]))
        output = tf.matmul(layer, weights[-1]) + biases[-1]
        pred = tf.nn.softmax(output)
        responses[response] = sess.run(pred, feed_dict={x: predictors})

print("Projections for {}:".format(symbol))
for key, value in responses.items():
    print('{} probability of price going up: {}%'.format(key, round(value[0,1]*100, 2)))

# Load regression model and make prediction about fututr stock price
graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    weights = []
    biases = []
    saver = tf.train.import_meta_graph('model.meta')
    saver.restore(sess, 'model')
    x = graph.get_tensor_by_name("x:0")
    for layer_num in range(100):
        try:
            weights.append(graph.get_tensor_by_name("weights_{}:0".format(layer_num)))
            biases.append(graph.get_tensor_by_name("biases_{}:0".format(layer_num)))
        except:
            num_layers = layer_num-1
            break
    layer = x
    for layer_num in range(num_layers):
        layer = tf.nn.relu(tf.add(tf.matmul(layer, weights[layer_num]), biases[layer_num]))
    output = tf.matmul(layer, weights[-1]) + biases[-1]
    with open('saved_models/' + 'daily' + '/' + 'predictors.json', 'r') as fd:
        predictor_names = json.loads(fd.read())
    predictors = np.column_stack(([variables_dict[variable_name] for variable_name in predictor_names]))
    prediction = sess.run(output, feed_dict={x: predictors})[0][0]
    print(data['Close'][-1] * (prediction/100+1))

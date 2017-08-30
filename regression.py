from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
from math import isnan
import os
import tensorflow as tf
from utils import create_network

r"""
This script is used to train neural networks to predict stock price for parameter optimization. Can be used to try out 
different metrics on different response variables, as well as try different network configurations.
"""

predictor_names = ['BIAS5', 'BIAS10', 'BIAS15', 'BIAS20', 'BIAS25', 'PSY5', 'PSY10', 'PSY15', 'PSY20', 'PSY25', 'ASY1',
                   'ASY2', 'ASY3', 'ASY4', 'ASY5', 'ASY6', 'ASY7', 'ASY8', 'ASY9', 'ASY10', 'ASY15', 'ASY20', 'ASY25']
current_predictors = ['BIAS5', 'PSY10', 'ASY1', 'ASY2', 'ASY3', 'ASY4', 'ASY5']
current_response = 'daily'
response_names = ['daily', 'weekly', 'bi_weekly', 'monthly']
variables_dict = {}

for name in predictor_names:
    variables_dict[name] = []
for name in response_names:
    variables_dict[name] = []

print('Beginning Download')

for folder_name in os.listdir('Raw_Stock_Data')[1:]:
    for file_name in os.listdir('Raw_Stock_Data/' + folder_name)[1:10]:
        data = pd.read_csv('Raw_Stock_Data/' + folder_name + '/' + file_name)
        MA5 = data['Close'].shift().rolling(window=5).mean()
        #MA10 = data['Close'].shift().rolling(window=10).mean()
        #MA15 = data['Close'].shift().rolling(window=15).mean()
        #MA20 = data['Close'].shift().rolling(window=20).mean()
        #MA25 = data['Close'].shift().rolling(window=25).mean()
        variables_dict['BIAS5'].extend(((data['Close'].shift() - MA5.shift()) / MA5.shift())[25:-25])
        #variables_dict['BIAS10'].extend(((data['Close'].shift() - MA10.shift()) / MA10.shift())[25:-25])
        #variables_dict['BIAS15'].extend(((data['Close'].shift() - MA15.shift()) / MA15.shift())[25:-25])
        #variables_dict['BIAS20'].extend(((data['Close'].shift() - MA20.shift()) / MA20.shift())[25:-25])
        #variables_dict['BIAS25'].extend(((data['Close'].shift() - MA25.shift()) / MA25.shift())[25:-25])
        positive = pd.Series(np.where(data['Close'] >= data['Open'], 1, 0))
        variables_dict['daily'].extend(((data['Close']-data['Close'].shift()) / data['Close'].shift() / .01)[25:-25])
        #variables_dict['weekly'].extend((pd.Series(np.where(data['Close'].shift(-5) >= data['Open'], 1, 0)))[25:-25])
        #variables_dict['bi_weekly'].extend((pd.Series(np.where(data['Close'].shift(-10) >= data['Open'], 1, 0)))[25:-25])
        #variables_dict['monthly'].extend((pd.Series(np.where(data['Close'].shift(-22) >= data['Open'], 1, 0)))[25:-25])
        #variables_dict['PSY5'].extend((positive.shift().rolling(window=5).sum())[25:-25])
        variables_dict['PSY10'].extend((positive.shift().rolling(window=10).sum())[25:-25])
        #variables_dict['PSY15'].extend((positive.shift().rolling(window=15).sum())[25:-25])
        #variables_dict['PSY20'].extend((positive.shift().rolling(window=20).sum())[25:-25])
        #variables_dict['PSY25'].extend((positive.shift().rolling(window=25).sum())[25:-25])
        SY = np.log(data['Close']) - np.log(data['Close'].shift())
        variables_dict['ASY1'].extend((SY.shift())[25:-25])
        variables_dict['ASY2'].extend((SY.shift().rolling(window=2).mean())[25:-25])
        variables_dict['ASY3'].extend((SY.shift().rolling(window=3).mean())[25:-25])
        variables_dict['ASY4'].extend((SY.shift().rolling(window=4).mean())[25:-25])
        variables_dict['ASY5'].extend((SY.shift().rolling(window=5).mean())[25:-25])
        #variables_dict['ASY6'].extend((SY.shift().rolling(window=6).mean())[25:-25])
        #variables_dict['ASY7'].extend((SY.shift().rolling(window=7).mean())[25:-25])
        #variables_dict['ASY8'].extend((SY.shift().rolling(window=8).mean())[25:-25])
        #variables_dict['ASY9'].extend((SY.shift().rolling(window=9).mean())[25:-25])
        #variables_dict['ASY10'].extend((SY.shift().rolling(window=10).mean())[25:-25])
        #variables_dict['ASY15'].extend((SY.shift().rolling(window=15).mean())[25:-25])
        #variables_dict['ASY20'].extend((SY.shift().rolling(window=20).mean())[25:-25])
        #variables_dict['ASY25'].extend((SY.shift().rolling(window=25).mean())[25:-25])

print('Finished Downloading Data')

idx_to_remove = []
for variable in ['BIAS5', 'PSY10', 'ASY5']:
    for idx, element in enumerate(variables_dict[variable]):
        if isnan(element):
            idx_to_remove.append(idx)
idx_to_remove = set(idx_to_remove)

print("Finished Checking for NaN Elements")

for key, value in variables_dict.items():
    if key in current_predictors or key in current_response:
        variables_dict[key] = [i for j, i in enumerate(value) if j not in idx_to_remove]

print('Finished Removing NaN Elements')

for variable_name in current_predictors:
    variables_dict[variable_name] = scale(variables_dict[variable_name])

print('Finished Scaling Variables')

predictors = np.column_stack(([variables_dict[variable_name] for variable_name in current_predictors]))

print('Finished Munging Data')

xtrain, xtest, ytrain, ytest = train_test_split(predictors, variables_dict[current_response], random_state=33)

print('Finished Splitting Data')

# Parameters
learning_rate = 0.01
training_epochs = 10
# Network Parameters
hidden_layer_nodes = [100, 100, 100]

# tf Graph input
x = tf.placeholder("float", [None, len(xtrain[0])], name='x')
y = tf.placeholder("float", [None])


# Construct model
pred = tf.transpose(create_network(x, hidden_layer_nodes, num_classes=1))

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(pred-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        # Run optimization op (backprop) and cost op (to get loss value)
        sess.run([optimizer, cost, pred], feed_dict={x: xtrain, y: ytrain})
        print('Completed Epoch {} of {}'.format(epoch + 1, training_epochs))

    print("Optimization Finished!")

    # Test model
    predicted_vals = sess.run(pred, feed_dict={x: xtest})
    # Calculate accuracy
    accuracy = sess.run(tf.reduce_mean(tf.square(predicted_vals-y)), feed_dict={x: xtest, y: ytest})
    print("Accuracy:", accuracy)

    #tf.train.Saver().save(sess, 'model')
"""
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
    print(sess.run(output, feed_dict={x: [predictors[1]]})[0][0])
"""
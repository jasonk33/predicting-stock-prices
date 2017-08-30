from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
from math import isnan, isinf
import os
import tensorflow as tf
from utils import create_network

r"""
This script is used to train neural networks to predict stock price direction for parameter optimization. Can be used to
try out different metrics on different response variables, as well as try different network configurations.
"""

all_predictors = ['BIAS5', 'BIAS10', 'BIAS15', 'BIAS20', 'BIAS25', 'PSY5', 'PSY10', 'PSY15', 'PSY20', 'PSY25', 'ASY1',
                   'ASY2', 'ASY3', 'ASY4', 'ASY5', 'ASY6', 'ASY7', 'ASY8', 'ASY9', 'ASY10', 'ASY15', 'ASY20', 'ASY25']

all_response = ['daily', 'weekly', 'bi_weekly', 'monthly']

predictor_names = ['BIAS5', 'PSY10', 'ASY5']

response = 'daily'

variables_dict = {}

for name in all_predictors + all_response:
    variables_dict[name] = []

print('Beginning Download')

for folder_name in os.listdir('Raw_Stock_Data')[1:]:
    for file_name in os.listdir('Raw_Stock_Data/' + folder_name)[1:10]:
        data = pd.read_csv('Raw_Stock_Data/' + folder_name + '/' + file_name)
        MA5 = data['Close'].shift().rolling(window=5).mean()
        """"
        MA10 = data['Close'].shift().rolling(window=10).mean()
        MA15 = data['Close'].shift().rolling(window=15).mean()
        MA20 = data['Close'].shift().rolling(window=20).mean()
        MA25 = data['Close'].shift().rolling(window=25).mean()
        """
        variables_dict['BIAS5'].extend(((data['Close'].shift() - MA5.shift())/MA5.shift())[25:-25])
        """
        variables_dict['BIAS10'].extend(((data['Close'].shift() - MA10.shift())/MA10.shift())[25:-25])
        variables_dict['BIAS15'].extend(((data['Close'].shift() - MA15.shift())/MA15.shift())[25:-25])
        variables_dict['BIAS20'].extend(((data['Close'].shift() - MA20.shift())/MA20.shift())[25:-25])
        variables_dict['BIAS25'].extend(((data['Close'].shift() - MA25.shift())/MA25.shift())[25:-25])
        """
        positive = pd.Series(np.where(data['Close'] >= data['Open'], 1, 0))
        variables_dict['daily'].extend(positive[25:-25])
        """
        variables_dict['weekly'].extend((pd.Series(np.where(data['Close'].shift(-5) >= data['Open'], 1, 0)))[25:-25])
        variables_dict['bi_weekly'].extend((pd.Series(np.where(data['Close'].shift(-10) >= data['Open'], 1, 0)))[25:-25])
        variables_dict['monthly'].extend((pd.Series(np.where(data['Close'].shift(-22) >= data['Open'], 1, 0)))[25:-25])
        variables_dict['PSY5'].extend((positive.shift().rolling(window=5).sum())[25:-25])
        """
        variables_dict['PSY10'].extend((positive.shift().rolling(window=10).sum())[25:-25])
        """
        variables_dict['PSY15'].extend((positive.shift().rolling(window=15).sum())[25:-25])
        variables_dict['PSY20'].extend((positive.shift().rolling(window=20).sum())[25:-25])
        variables_dict['PSY25'].extend((positive.shift().rolling(window=25).sum())[25:-25])
        """
        SY = np.log(data['Close']) - np.log(data['Close'].shift())
        """
        variables_dict['ASY1'].extend((SY.shift())[25:-25])
        variables_dict['ASY2'].extend((SY.shift().rolling(window=2).mean())[25:-25])
        variables_dict['ASY3'].extend((SY.shift().rolling(window=3).mean())[25:-25])
        variables_dict['ASY4'].extend((SY.shift().rolling(window=4).mean())[25:-25])
        """
        variables_dict['ASY5'].extend((SY.shift().rolling(window=5).mean())[25:-25])
        """
        variables_dict['ASY6'].extend((SY.shift().rolling(window=6).mean())[25:-25])
        variables_dict['ASY7'].extend((SY.shift().rolling(window=7).mean())[25:-25])
        variables_dict['ASY8'].extend((SY.shift().rolling(window=8).mean())[25:-25])
        variables_dict['ASY9'].extend((SY.shift().rolling(window=9).mean())[25:-25])
        variables_dict['ASY10'].extend((SY.shift().rolling(window=10).mean())[25:-25])
        variables_dict['ASY15'].extend((SY.shift().rolling(window=15).mean())[25:-25])
        variables_dict['ASY20'].extend((SY.shift().rolling(window=20).mean())[25:-25])
        variables_dict['ASY25'].extend((SY.shift().rolling(window=25).mean())[25:-25])
"""

print('Finished Downloading Data')

idx_to_remove = []
for variable in predictor_names:
    for idx, element in enumerate(variables_dict[variable]):
        if isnan(element) or isinf(element):
            idx_to_remove.append(idx)
idx_to_remove = set(idx_to_remove)

print("Finished Checking for NaN Elements")

for key, value in variables_dict.items():
    if key in predictor_names or key == response:
        variables_dict[key] = [i for j, i in enumerate(value) if j not in idx_to_remove]

print('Finished Removing NaN Elements')

for variable_name in predictor_names:
    variables_dict[variable_name] = scale(variables_dict[variable_name])

print('Finished Scaling Variables')

predictors = np.column_stack(([variables_dict[variable_name] for variable_name in predictor_names]))

print('Finished Munging Data')

xtrain, xtest, ytrain, ytest = train_test_split(predictors, variables_dict[response], random_state=33)

print('Finished Splitting Data')
ytrain = np.where(np.array(ytrain) >= 0, 1, 0)
ytest = np.where(np.array(ytest) >= 0, 1, 0)
ytrain_hot = np.zeros((len(ytrain), 2))
ytrain_hot[np.arange(len(ytrain)), ytrain] = 1

ytest_hot = np.zeros((len(ytest), 2))
ytest_hot[np.arange(len(ytest)), ytest] = 1

x = tf.placeholder(tf.float32, [None, len(predictor_names)], name='x')
y = tf.placeholder(tf.float32, [None, 2])

output_layer = create_network(x, [33, 33, 33], num_classes=2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_layer))
train_step = tf.train.AdamOptimizer(.01).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


print('Starting Model Training')

num_epochs = 33
for epoch in range(num_epochs):
    sess.run(train_step, feed_dict={x: xtrain, y: ytrain_hot})
    print('Completed Epoch {} of {}'.format(epoch + 1, num_epochs))

correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print("Model Accuracy: {}% -- Naive Model: {}%".format(sess.run(accuracy, feed_dict={x: xtest, y: ytest_hot})*100,
                                                       np.mean(ytest)*100))

sess.close()
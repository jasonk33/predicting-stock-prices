from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
from math import isnan
import os
import tensorflow as tf
import json

predictor_names = ['BIAS5', 'BIAS10', 'BIAS15', 'BIAS20', 'BIAS25', 'PSY5', 'PSY10', 'PSY15', 'PSY20', 'PSY25', 'ASY1',
                   'ASY2', 'ASY3', 'ASY4', 'ASY5', 'ASY6', 'ASY7', 'ASY8', 'ASY9', 'ASY10', 'ASY15', 'ASY20', 'ASY25']
current_predictors = ['BIAS5', 'PSY10', 'ASY1', 'ASY2', 'ASY3', 'ASY4', 'ASY5']
response_names = ['daily', 'weekly', 'bi_weekly', 'monthly']
variables_dict = {}
"""
for name in predictor_names:
    variables_dict[name] = []
for name in response_names:
    variables_dict[name] = []

print('Beginning Download')

for folder_name in os.listdir('/Users/JasonKatz/Desktop/Code/Stocks/Raw_Stock_Data')[1:]:
    for file_name in os.listdir('/Users/JasonKatz/Desktop/Code/Stocks/Raw_Stock_Data/' + folder_name)[1:]:
        data = pd.read_csv('/Users/JasonKatz/Desktop/Code/Stocks/Raw_Stock_Data/' + folder_name + '/' + file_name)
        MA5 = data['Close'].shift().rolling(window=5).mean()
        MA10 = data['Close'].shift().rolling(window=10).mean()
        MA15 = data['Close'].shift().rolling(window=15).mean()
        MA20 = data['Close'].shift().rolling(window=20).mean()
        MA25 = data['Close'].shift().rolling(window=25).mean()
        variables_dict['BIAS5'].extend(((data['Close'].shift() - MA5.shift())/MA5.shift())[25:-25])
        variables_dict['BIAS10'].extend(((data['Close'].shift() - MA5.shift())/MA10.shift())[25:-25])
        variables_dict['BIAS15'].extend(((data['Close'].shift() - MA5.shift())/MA15.shift())[25:-25])
        variables_dict['BIAS20'].extend(((data['Close'].shift() - MA5.shift())/MA20.shift())[25:-25])
        variables_dict['BIAS25'].extend(((data['Close'].shift() - MA5.shift())/MA25.shift())[25:-25])
        positive = pd.Series(np.where(data['Close'] >= data['Open'], 1, 0))
        variables_dict['daily'].extend(positive[25:-25])
        variables_dict['weekly'].extend((pd.Series(np.where(data['Close'].shift(-5) >= data['Open'], 1, 0)))[25:-25])
        variables_dict['bi_weekly'].extend((pd.Series(np.where(data['Close'].shift(-10) >= data['Open'], 1, 0)))[25:-25])
        variables_dict['monthly'].extend((pd.Series(np.where(data['Close'].shift(-22) >= data['Open'], 1, 0)))[25:-25])
        variables_dict['PSY5'].extend((positive.shift().rolling(window=5).sum())[25:-25])
        variables_dict['PSY10'].extend((positive.shift().rolling(window=10).sum())[25:-25])
        variables_dict['PSY15'].extend((positive.shift().rolling(window=15).sum())[25:-25])
        variables_dict['PSY20'].extend((positive.shift().rolling(window=20).sum())[25:-25])
        variables_dict['PSY25'].extend((positive.shift().rolling(window=25).sum())[25:-25])
        SY = np.log(data['Close']) - np.log(data['Close'].shift())
        variables_dict['ASY1'].extend((SY.shift())[25:-25])
        variables_dict['ASY2'].extend((SY.shift().rolling(window=2).mean())[25:-25])
        variables_dict['ASY3'].extend((SY.shift().rolling(window=3).mean())[25:-25])
        variables_dict['ASY4'].extend((SY.shift().rolling(window=4).mean())[25:-25])
        variables_dict['ASY5'].extend((SY.shift().rolling(window=5).mean())[25:-25])
        variables_dict['ASY6'].extend((SY.shift().rolling(window=6).mean())[25:-25])
        variables_dict['ASY7'].extend((SY.shift().rolling(window=7).mean())[25:-25])
        variables_dict['ASY8'].extend((SY.shift().rolling(window=8).mean())[25:-25])
        variables_dict['ASY9'].extend((SY.shift().rolling(window=9).mean())[25:-25])
        variables_dict['ASY10'].extend((SY.shift().rolling(window=10).mean())[25:-25])
        variables_dict['ASY15'].extend((SY.shift().rolling(window=15).mean())[25:-25])
        variables_dict['ASY20'].extend((SY.shift().rolling(window=20).mean())[25:-25])
        variables_dict['ASY25'].extend((SY.shift().rolling(window=25).mean())[25:-25])


print('Finished Downloading Data')

idx_to_remove = []
for variable in ['BIAS25', 'PSY25', 'ASY25']:
    for idx, element in enumerate(variables_dict[variable]):
        if isnan(element):
            idx_to_remove.append(idx)
idx_to_remove = set(idx_to_remove)

print("Finished Checking for NaN Elements")

for key, value in variables_dict.items():
    if key in predictor_names or key in response_names:
        variables_dict[key] = [i for j, i in enumerate(value) if j not in idx_to_remove]

print('Finished Removing NaN Elements')

for variable_name in predictor_names:
    variables_dict[variable_name] = scale(variables_dict[variable_name])

print('Finished Scaling Variables')

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

print('Beginning json dump')

counter = 0
variabls_to_write = len(variables_dict)
for key, value in variables_dict.items():
    counter += 1
    with open('json_files/' + key + '.json', 'w') as fd:
        fd.write(json.dumps(value, indent=4, cls=MyEncoder))
    print("Finished writing to json variable {} of {}".format(counter, variabls_to_write))
"""
num_variables = len(current_predictors)+1
counter = 0
print('Beginning Download')
for predictor in current_predictors + ['daily']:
    counter += 1
    with open('json_files/' + predictor + '.json', 'r') as fd:
        variables_dict[predictor] = json.loads(fd.read())
    print("Finished downloading json variable {} of {}".format(counter, num_variables))

predictors = np.column_stack(([variables_dict[variable_name] for variable_name in current_predictors]))

print('Finished Munging Data')

xtrain, xtest, ytrain, ytest = train_test_split(predictors, variables_dict['daily'])

print('Finished Splitting Data')

ytrain_hot = np.zeros((len(ytrain), 2))
ytrain_hot[np.arange(len(ytrain)), ytrain] = 1

ytest_hot = np.zeros((len(ytest), 2))
ytest_hot[np.arange(len(ytest)), ytest] = 1

x = tf.placeholder(tf.float32, [None, len(current_predictors)], name='x')
W = tf.Variable(tf.zeros([len(current_predictors), 2]), name='W')
b = tf.Variable(tf.zeros([2]), name='b')
y = tf.add(tf.matmul(x, W), b)
#W2 = tf.Variable(tf.zeros([12, 2]), name='W2')
#b2 = tf.Variable(tf.zeros([2]), name='b2')
#y2 = tf.matmul(y, W2)
y_ = tf.placeholder(tf.float32, [None, 2])

saver = tf.train.Saver()
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


print('Starting Model Training')

for _ in range(25):
    sess.run(train_step, feed_dict={x: xtrain, y_: ytrain_hot})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print("Model Accuracy: {}% -- Naive Model: {}%".format(sess.run(accuracy, feed_dict={x: xtest, y_: ytest_hot})*100,
                                                       np.mean(ytest)*100))
                                                       
saver.save(sess, 'my_test_model')
"""
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('my_test_model.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    W = graph.get_tensor_by_name("W:0")
    b = graph.get_tensor_by_name("b:0")
    output = tf.add(tf.matmul(x, W), b)
    pred = tf.nn.softmax(output)
    print(sess.run(pred, feed_dict={x: xtest[0:1]}))
"""
"""
n_nodes_hl1 = 15
n_nodes_hl2 = 15
n_nodes_hl3 = 15

n_classes = 2

# height x width
x = tf.placeholder('float',[None, len(current_predictors)])
y = tf.placeholder('float')

def neural_network_model(data):

	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(current_predictors), n_nodes_hl1])),
					   'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					   'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					   'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					   'biases':tf.Variable(tf.random_normal([n_classes]))}

	# (input_data * weights) + biases

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	#l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	#l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	#l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			epoch_x, epoch_y = xtrain, ytrain_hot
			_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
			epoch_loss += c
			print('Epoch', epoch+1, 'completed out of', hm_epochs,'loss:',epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:',accuracy.eval({x:xtest, y:ytest_hot}))

train_neural_network(x)
print(np.mean(ytest))
train_neural_network(x)
print(np.mean(ytest))
train_neural_network(x)
print(np.mean(ytest))

n_nodes_hl1 = 50
n_nodes_hl2 = 50
n_nodes_hl3 = 50

train_neural_network(x)
print(np.mean(ytest))
train_neural_network(x)
print(np.mean(ytest))
train_neural_network(x)
print(np.mean(ytest))
"""

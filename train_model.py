import tensorflow as tf
from utils import load_from_json, create_model_data

"""
Available Predictors: 
    BIAS: 5, 10, 15, 20, 25
    PSY: 5, 10, 15, 20, 25
    ASY: 1:10, 15, 20, 25
Available Responses:
    daily, weekly, bi_weekly, monthly
"""
predictors = ['BIAS5', 'PSY10', 'ASY1', 'ASY2', 'ASY3', 'ASY4', 'ASY5']

response = 'daily'

variables_dict = load_from_json(predictors, response)

xtrain, xtest, ytrain, ytest = create_model_data(variables_dict, predictors, response)

x = tf.placeholder(tf.float32, [None, len(predictors)], name='x')
weights = tf.Variable(tf.zeros([len(predictors), 2]), name='weights')
biases = tf.Variable(tf.zeros([2]), name='biases')
y = tf.add(tf.matmul(x, weights), biases)
y_ = tf.placeholder(tf.float32, [None, 2])

saver = tf.train.Saver()
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


print('Starting Model Training')

for _ in range(25):
    sess.run(train_step, feed_dict={x: xtrain, y_: ytrain})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print("Model Accuracy: {}% -- Naive Model: {}%".format(sess.run(accuracy, feed_dict={x: xtest, y_: ytest})*100,
                                                       ytest[:, 1].mean()*100))
                                                       
saver.save(sess, 'my_test_model')

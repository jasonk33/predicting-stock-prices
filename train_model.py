import tensorflow as tf
import json
from utils import load_from_json, create_model_data, create_network

"""
Available Predictors: 
    BIAS: 5, 10, 15, 20, 25
    PSY: 5, 10, 15, 20, 25
    ASY: 1:10, 15, 20, 25
Available Responses:
    daily, weekly, bi_weekly, monthly
"""
num_epochs = 10

hidden_layers = [25, 25]

learning_rate = .01

for response in ['daily', 'weekly', 'bi_weekly', 'monthly']:
    print('Beginning {} Models'.format(response))

    with open('saved_models/' + response + '/' + 'predictors.json', 'r') as fd:
        predictors = json.loads(fd.read())

    variables_dict = load_from_json(predictors, response, verbose=True)

    xtrain, xtest, ytrain, ytest, ytrain_hot, ytest_hot = create_model_data(variables_dict, predictors, response,
                                                                            model_type='Both')
    for model_type in ['Regression', 'Classification']:
        print('Beginning {} Model for {} Predictions'.format(model_type, response))
        print('Setting up TensorBoard')
        x = tf.placeholder(tf.float32, [None, len(predictors)], name='x')
        if model_type == 'Classification':
            y = tf.placeholder(tf.float32, [None, 2], name='y')
            output_layer = create_network(x, hidden_layers, num_classes=2)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_layer))
        else:
            y = tf.placeholder(tf.float32, name='y')
            output_layer = tf.transpose(create_network(x, hidden_layers, num_classes=1))
            cost = tf.reduce_mean(tf.square(output_layer-y))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('Starting Model Training')
            for epoch in range(num_epochs):
                if model_type == 'Classification':
                    sess.run(optimizer, feed_dict={x: xtrain, y: ytrain_hot})
                else:
                    sess.run(optimizer, feed_dict={x: xtrain, y: ytrain})
                print('Completed Epoch {} of {}'.format(epoch + 1, num_epochs))

            if model_type == 'Classification':
                correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                model_accuracy = sess.run(accuracy, feed_dict={x: xtest, y: ytest_hot}) * 100
                naive_accuracy = ytest_hot[:, 1].mean() * 100
                print("Model Accuracy: {}% -- Naive Model: {}%".format(model_accuracy, naive_accuracy))
                with open('saved_models/' + response + '/' + model_type + '/accuracy.json', 'w') as fd:
                    fd.write(json.dumps({'Model': model_accuracy, 'Naive': naive_accuracy}, indent=4))
            else:
                predicted_vals = sess.run(output_layer, feed_dict={x: xtest})
                model_accuracy = sess.run(tf.reduce_mean(tf.square(predicted_vals - y)), feed_dict={x: xtest, y: ytest})
                print("Sum of Squares:", model_accuracy)
            print('Saving Model')
            tf.train.Saver().save(sess, 'saved_models/' + response + '/' + model_type + '/model')
            print('Finished {} Model for {} Predictions'.format(model_type, response))

print('Finished Training and Saving Models')

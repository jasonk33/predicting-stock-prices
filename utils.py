import json
import numpy as np
from math import isnan
from sklearn.model_selection import train_test_split
import tensorflow as tf


class MyEncoder(json.JSONEncoder):
    r"""
    Encoder class to convert object to proper format for json uploading
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def check_for_nan_elements(variables_dict, variables_to_check, verbose=False):
    r"""
    A method for checking which trading days have metrics that contain NaN elements

    Parameters
    ----------
    variables_dict : dict
        Dictionary that contains the metric names and their values
    variables_to_check : list[str]
        List of the metrics to check for NaN elements
    verbose : bool, optional
        Print progress of method

    Returns
    -------
    idx_to_remove : list[int]
        The set of indices that contain NaN values
    """
    idx_to_remove = []
    num_variables = len(variables_to_check)
    counter = 0
    for variable in variables_to_check:
        counter += 1
        for idx, element in enumerate(variables_dict[variable]):
            if isnan(element):
                idx_to_remove.append(idx)
        if verbose:
            print("Finished Checking NaN for Variable {} of {}".format(counter, num_variables))
    print("Finished Checking for NaN Elements")
    return set(idx_to_remove)


def remove_nan_elements(variables_dict, idx_to_remove, verbose=False):
    r"""
    A method for removing trading days with metrics containing NaN elements

    Parameters
    ----------
    variables_dict : dict
        Dictionary that contains the metric names and their values
    idx_to_remove : list[int]
        List of indices to be removed
    verbose : bool, optional
        Print progress of method
    """
    num_variables = len(variables_dict)
    counter = 0
    for key, value in variables_dict.items():
        counter += 1
        variables_dict[key] = [i for j, i in enumerate(value) if j not in idx_to_remove]
        if verbose:
            print("Finished Removing NaN for Variable {} of {}".format(counter, num_variables))
    print('Finished Removing NaN Elements')


def write_to_json(variables_dict, verbose=False):
    r"""
    A method for writing to json all metrics calculated

    Parameters
    ----------
    variables_dict : dict
        Dictionary that contains the metric names and their values
    verbose : bool, optional
        Print progress of method
    """
    print('Beginning json dump')
    counter = 0
    num_variables = len(variables_dict)
    for key, value in variables_dict.items():
        counter += 1
        with open('json_files/' + key + '.json', 'w') as fd:
            fd.write(json.dumps(value, indent=4, cls=MyEncoder))
        if verbose:
            print("Finished writing to json variable {} of {}".format(counter, num_variables))


def load_from_json(predictors, response, verbose=False):
    r"""
    A method for loading all metrics from file

    Parameters
    ----------
    predictors : list[str]
        List of the metrics
    response : list[str]
        List of the response variables
    verbose : bool, optional
        Print progress of method

    Returns
    -------
    variables_dict : dict
        Dictionary that contains the metric names and their values
    """
    variables_dict = {}
    num_variables = len(predictors) + 1
    counter = 0
    print('Beginning Download')
    for predictor in predictors + [response]:
        counter += 1
        with open('json_files/' + predictor + '.json', 'r') as fd:
            variables_dict[predictor] = json.loads(fd.read())
        if verbose:
            print("Finished downloading json variable {} of {}".format(counter, num_variables))
    return variables_dict


def create_model_data(variables_dict, predictors, response, model_type='Both'):
    r"""
    A method for preparing data into train and test sets to feed into a model

    Parameters
    ----------
    variables_dict : dict
        Dictionary that contains the metric names and their values
    predictors : list[str]
        List of the metrics
    response : list[str]
        List of the response variables
    model_type : str, optional
        Either 'Classification', 'Regression', or 'Both'

    Returns
    -------
    xtrain : array
        Predictor values for training
    xtest : array
        Predictor values for testing
    ytrain : array
        Response values for training regression
    ytest : array
        Response values for testing regression
    ytrain_hot : array, optional
        Response values for training classification
    ytest_hot : array
        Response values for testing classification
    """
    predictors = np.column_stack(([variables_dict[variable_name] for variable_name in predictors]))
    print('Finished Munging Data')
    xtrain, xtest, ytrain, ytest = train_test_split(predictors, variables_dict[response], random_state=33)
    print('Finished Splitting Data')
    if model_type == 'Regression':
        return xtrain, xtest, ytrain, ytest
    ytrain_new = np.where(np.array(ytrain) >= 0, 1, 0)
    ytest_new = np.where(np.array(ytest) >= 0, 1, 0)
    ytrain_hot = np.zeros((len(ytrain_new), 2))
    ytrain_hot[np.arange(len(ytrain_new)), ytrain_new] = 1
    ytest_hot = np.zeros((len(ytest_new), 2))
    ytest_hot[np.arange(len(ytest_new)), ytest_new] = 1
    if model_type == 'Classification':
        return xtrain, xtest, ytrain_hot, ytest_hot
    else:
        return xtrain, xtest, ytrain, ytest, ytrain_hot, ytest_hot


def scale_and_save(variables_dict, predictor_names, verbose=False):
    r"""
    A method for scaling predictor values and saving the scaling parameters to file

    Parameters
    ----------
    variables_dict : dict
        Dictionary that contains the metric names and their values
    predictor_names : list[str]
        List of the metrics
    verbose : bool, optional
        Print progress of method
    """
    scaling_params = {}
    num_variables = len(predictor_names)
    counter = 0
    for variable_name in predictor_names:
        counter += 1
        mean = np.mean(variables_dict[variable_name])
        std = np.std(variables_dict[variable_name])
        variables_dict[variable_name] = (variables_dict[variable_name] - mean) / std
        scaling_params[variable_name] = {'mean': mean, 'std': std}
        if verbose:
            print("Finished Scaling Variable {} of {}".format(counter, num_variables))
    with open('scaling_params.json', 'w') as fd:
        fd.write(json.dumps(scaling_params, indent=4))
    print('Finished Scaling Variables')


def scale_individual_stock(variables_dict):
    r"""
    A method for scaling an individual stock's predictor values using scaling parameters saved locally

    Parameters
    ----------
    variables_dict : dict
        Dictionary that contains the metric names and their values
    """
    with open('scaling_params.json', 'r') as fd:
        scaling_params = json.loads(fd.read())
    for key, value in variables_dict.items():
        if 'MA' not in key:
            mean = scaling_params[key]['mean']
            std = scaling_params[key]['std']
            variables_dict[key] = (value - mean) / std

def create_network(x, nodes=[], num_classes=1):
    r"""
    A method for building a neural network with a variable number of hidden layers

    Parameters
    ----------
    nodes : list[int], optional
        How many nodes each of hidden layer will contain
    num_classes : int, optional
        Number of ouput nodes (1 for regression, more than 1 for classification)

    Returns
    -------
    output_layer : TensorFlow object
        The output layer of the constructed neural network
    """
    nodes.insert(0, x.get_shape().as_list()[1])
    nodes.append(num_classes)
    weights = []
    biases = []
    layer = x
    for layer_num, num_nodes in enumerate(nodes[:-1]):
        weights.append(tf.Variable(tf.random_normal([num_nodes, nodes[layer_num+1]], 0, 0.1), name="weights_{}".format(layer_num)))
        biases.append(tf.Variable(tf.random_normal([nodes[layer_num+1]], 0, 0.1), name="biases_{}".format(layer_num)))
        if layer_num > 0:
            layer = tf.nn.relu(tf.add(tf.matmul(layer, weights[layer_num-1]), biases[layer_num-1]))
    return tf.matmul(layer, weights[-1]) + biases[-1]

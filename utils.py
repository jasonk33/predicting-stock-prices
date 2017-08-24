import json
import numpy as np
from math import isnan
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split


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


def check_for_nan_elements(variables_dict, variables_to_check):
    idx_to_remove = []
    for variable in variables_to_check:
        for idx, element in enumerate(variables_dict[variable]):
            if isnan(element):
                idx_to_remove.append(idx)
    print("Finished Checking for NaN Elements")
    return set(idx_to_remove)


def remove_nan_elements(variables_dict, idx_to_remove):
    for key, value in variables_dict.items():
        variables_dict[key] = [i for j, i in enumerate(value) if j not in idx_to_remove]
    print('Finished Removing NaN Elements')


def scale_predictors(variables_dict, predictor_names):
    for variable_name in predictor_names:
        variables_dict[variable_name] = scale(variables_dict[variable_name])
    print('Finished Scaling Variables')


def write_to_json(variables_dict):
    print('Beginning json dump')
    counter = 0
    num_variables = len(variables_dict)
    for key, value in variables_dict.items():
        counter += 1
        with open('json_files/' + key + '.json', 'w') as fd:
            fd.write(json.dumps(value, indent=4, cls=MyEncoder))
        print("Finished writing to json variable {} of {}".format(counter, num_variables))


def load_from_json(predictors, response):
    variables_dict = {}
    num_variables = len(predictors) + 1
    counter = 0
    print('Beginning Download')
    for predictor in predictors + [response]:
        counter += 1
        with open('json_files/' + predictor + '.json', 'r') as fd:
            variables_dict[predictor] = json.loads(fd.read())
        print("Finished downloading json variable {} of {}".format(counter, num_variables))
    return variables_dict


def create_model_data(variables_dict, predictors, response):
    predictors = np.column_stack(([variables_dict[variable_name] for variable_name in predictors]))
    print('Finished Munging Data')
    xtrain, xtest, ytrain, ytest = train_test_split(predictors, variables_dict[response])
    ytrain_hot = np.zeros((len(ytrain), 2))
    ytrain_hot[np.arange(len(ytrain)), ytrain] = 1
    ytest_hot = np.zeros((len(ytest), 2))
    ytest_hot[np.arange(len(ytest)), ytest] = 1
    print('Finished Splitting Data')
    return xtrain, xtest, ytrain_hot, ytest_hot

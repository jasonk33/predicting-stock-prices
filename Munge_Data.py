import numpy as np
import pandas as pd
import os
from utils import check_for_nan_elements, remove_nan_elements, write_to_json, scale_and_save

r"""
This script uses raw stock data to calculate various performance metrics and save that data to file.
"""

# Predictor names and response names
predictor_names = ['BIAS5', 'BIAS10', 'BIAS15', 'BIAS20', 'BIAS25', 'PSY5', 'PSY10', 'PSY15', 'PSY20', 'PSY25', 'ASY1',
                   'ASY2', 'ASY3', 'ASY4', 'ASY5', 'ASY6', 'ASY7', 'ASY8', 'ASY9', 'ASY10', 'ASY15', 'ASY20', 'ASY25']
response_names = ['daily', 'weekly', 'bi_weekly', 'monthly']

variables_dict = {}
for name in predictor_names:
    variables_dict[name] = []
for name in response_names:
    variables_dict[name] = []

# Iterate through each file with raw stock data, calculate metrics, and append to current results
print('Beginning Download')
for folder_name in os.listdir('Raw_Stock_Data')[1:]:
    for file_name in os.listdir('Raw_Stock_Data/' + folder_name)[1:]:
        data = pd.read_csv('Raw_Stock_Data/' + folder_name + '/' + file_name)
        MA5 = data['Close'].shift().rolling(window=5).mean()
        MA10 = data['Close'].shift().rolling(window=10).mean()
        MA15 = data['Close'].shift().rolling(window=15).mean()
        MA20 = data['Close'].shift().rolling(window=20).mean()
        MA25 = data['Close'].shift().rolling(window=25).mean()
        variables_dict['BIAS5'].extend(((data['Close'].shift() - MA5.shift())/MA5.shift())[25:-25])
        variables_dict['BIAS10'].extend(((data['Close'].shift() - MA10.shift())/MA10.shift())[25:-25])
        variables_dict['BIAS15'].extend(((data['Close'].shift() - MA15.shift())/MA15.shift())[25:-25])
        variables_dict['BIAS20'].extend(((data['Close'].shift() - MA20.shift())/MA20.shift())[25:-25])
        variables_dict['BIAS25'].extend(((data['Close'].shift() - MA25.shift())/MA25.shift())[25:-25])
        positive = pd.Series(np.where(data['Close'] >= data['Close'].shift(), 1, 0))
        variables_dict['daily'].extend(((data['Close'] - data['Close'].shift()) / data['Close'].shift() / .01)[25:-25])
        variables_dict['weekly'].extend(((data['Close'] - data['Close'].shift(-5)) / data['Close'].shift(-5) / .01)[25:-25])
        variables_dict['bi_weekly'].extend(((data['Close'] - data['Close'].shift(-10)) / data['Close'].shift(-10) / .01)[25:-25])
        variables_dict['monthly'].extend(((data['Close'] - data['Close'].shift(-22)) / data['Close'].shift(-22) / .01)[25:-25])
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

# Clean and save data
idx_to_remove = check_for_nan_elements(variables_dict, predictor_names, verbose=True)
remove_nan_elements(variables_dict, idx_to_remove, verbose=True)
scale_and_save(variables_dict, predictor_names, verbose=True)
write_to_json(variables_dict, verbose=True)

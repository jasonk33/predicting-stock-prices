# Predicting Stock Price Direction using Deep Neural Networks

This repository contains code to download and munge historical stock data and construct neural networks to make predictions with this data

Historical data from ~3000 stocks were downloaded through Google Finance, code can be found in `Download_Raw_Data.py`

Momentum based performance metrics were calculated, this code can be found in `Munge_Data.py`

The models were constructed using Tensorflow, this code can be found in `train_model.py`

An individual stock price can be predicted from the saved models, this code can be found in `predict_stock.py`

Predictions have two forms
```
    - Probability the stock will increase in price over a given time period
    - Predicted stock price over a given time period in the future
```
There are four different time periods modelled:
```
    - One day
    - One week
    - Two weeks
    - One month
```

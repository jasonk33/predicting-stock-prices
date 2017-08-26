from pandas_datareader import data
from finsymbols import symbols

all_stocks = symbols.get_nyse_symbols() + symbols.get_nasdaq_symbols()

for stock in all_stocks:
    try:
        symbol = stock['symbol']
        historical_data = data.DataReader(symbol, 'Google', '2002-08-01', '2017-08-01')
        if len(historical_data) > 252:
            historical_data.to_csv('/Users/JasonKatz/Desktop/Code/Stocks/Raw_Stock_Data/' + symbol + '_data')
    except:
        pass

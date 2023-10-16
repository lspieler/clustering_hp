import pandas as pd
import numpy as np

def get_data(start, end, freq_per_second = 1000000000, directory = "data/", orderbook_filename = "orderbook_1.csv", message_filename = "message_1.csv"):
    df = _read_csv(start, end, freq_per_second, directory + orderbook_filename, directory + message_filename)
    df = _add_price(df)
    return df

def _read_csv(start, end, freq_per_second, orderbook_filename, message_filename):
    rows = end-start
    if rows <= 0:
        rows = None
    df = pd.read_csv(f'{orderbook_filename}', na_values=['nan'], skiprows=start, nrows=rows, header=None)
    times = pd.read_csv(f'{message_filename}', na_values=['nan'], skiprows=start, nrows=rows, usecols=[0], header=None).squeeze()
    # print lenght of times and df 
    df.set_index(times, inplace=True)
    # add date from filename to index and convert to datetime
    df.index = pd.to_datetime(df.index, unit='s')
    #resample index to freq_per_second
    df = df.resample(f'{freq_per_second}').mean()
    
    df.index.name = 'time'
    df.columns = ['a_price_0', 'a_size_0', 'b_price_0', 'b_size_0', 'a_price_1', 'a_size_1', 'b_price_1', 'b_size_1', 
                  'a_price_2', 'a_size_2', 'b_price_2', 'b_size_2', 'a_price_3', 'a_size_3', 'b_price_3', 'b_size_3', 
                  'a_price_4', 'a_size_4', 'b_price_4', 'b_size_4', 'a_price_5', 'a_size_5', 'b_price_5', 'b_size_5', 
                  'a_price_6', 'a_size_6', 'b_price_6', 'b_size_6', 'a_price_7', 'a_size_7', 'b_price_7', 'b_size_7', 
                  'a_price_8', 'a_size_8', 'b_price_8', 'b_size_8', 'a_price_9', 'a_size_9', 'b_price_9', 'b_size_9']
    return df

def _add_price(df):
    # Use bid-ask midpoint as price. Price is 10,000 times the actual price in the data.
    df['price'] = (df['b_price_0'] + df['a_price_0']) / 2 / 10000
    return df
import pandas as pd

def get_trades(df):
    trades = pd.Series(0, index=df.index)
    # Trade based on ask size and bid size:
    trades[df["a_size_0"] < df["b_size_0"]] = 1000
    trades[df["a_size_0"] > df["b_size_0"]] = -1000
    # Oracle:
    # trades[df["price"] < df.shift(-1)["price"]] = 1000
    # trades[df["price"] > df.shift(-1)["price"]] = -1000
    trades.iloc[-1] = 0
    trades.iloc[1:] = trades.diff().iloc[1:]
    return trades

def get_oracle_trades(df):
    trades = pd.Series(0, index=df.index)
    trades[df["price"] < df.shift(-1)["price"]] = 1000
    trades[df["price"] > df.shift(-1)["price"]] = -1000
    trades.iloc[-1] = 0
    trades.iloc[1:] = trades.diff().iloc[1:]
    return trades
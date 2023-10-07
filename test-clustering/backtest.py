import pandas as pd

def backtest(data, trades, starting_value = 200000, fixed_cost = 0.00, floating_cost = 0.00):
    
   #TODO: make this less shit
   
   data['CASH'] = 1.0

   daily_portfolio_values = pd.Series(0, index=data.index)
   trades_full = pd.DataFrame(index=data.index)
   trades_full['CASH'] = (data['price'] * trades) * -1
   trades_full[trades_full['CASH'] < 0] *= (1 + floating_cost)
   trades_full[trades_full['CASH'] > 0] *= (1 - floating_cost)
   trades_full[trades_full['CASH'] != 0] -= fixed_cost
   trades_full['CASH'].iat[0] += starting_value
   trades_full['CASH'] = trades_full['CASH'].cumsum()
   positions = trades.cumsum()
   trades_full['INVESTED'] = positions * data['price']

   daily_portfolio_values = trades_full.sum(axis=1)

   return daily_portfolio_values

def backtest_baseline(data, starting_value = 200000, fixed_cost = 0.00, floating_cost = 0.00):
   trades = pd.Series(0, index=data.index)
   trades.iloc[0] = 1000
   trades.iloc[-1] = -1000
   return backtest(data, trades, starting_value, fixed_cost, floating_cost)
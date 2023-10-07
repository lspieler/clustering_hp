import numpy as np
import pandas as pd
from get_data import get_data
from strategy import get_trades, get_oracle_trades
from backtest import backtest, backtest_baseline
import learners
import warnings
import datetime
from FFNN import FFNN
import matplotlib.pyplot as plt
import learners
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import glob


# Why do we need square loss? We don't use it.
# How to store weights well
# Multi-action learning (the efficiency thing)
# Baseline?
# Why do the activations functions tend to hate negative numbers?
# Big network breaky da shit

FREQ = 1000
FREQSTR = "1000_C"
FLOATING = 0.00001
START = 100000
END = 110000
GAMMA = 0.8


def DNE():
    df = get_data(0, 0, freq_per_second = FREQ, directory = "data/", orderbook_filename = "orderbook_1.csv", message_filename = "message_1.csv")
    df = df.iloc[:-1]
    trades = get_trades(df)
    daily_values = backtest(df, trades, floating_cost=FLOATING)
    baseline_values = backtest_baseline(df, floating_cost=FLOATING)
    print("\nFinal Portfolio Value: " + str(daily_values.iloc[-1]))
    print("Net Value: " + str(daily_values.iloc[-1] - 200000))
    print("Cumulative Returns: " + str(daily_values.iloc[-1] / daily_values.iloc[0] - 1))
    print("Baseline Cumulative Returns: " + str(baseline_values.iloc[-1] / baseline_values.iloc[0] - 1))
    print("Baseline Net Value: " + str(baseline_values.iloc[-1] - 200000))

def ORACLE():
    df = get_data(0, 0, freq_per_second = FREQ, directory = "data/", orderbook_filename = "orderbook_1.csv", message_filename = "message_1.csv")
    df = df.iloc[:-1]
    trades = get_oracle_trades(df)
    daily_values = backtest(df, trades, floating_cost=FLOATING)
    baseline_values = backtest_baseline(df, floating_cost=FLOATING)
    print("\nFinal Portfolio Value: " + str(daily_values.iloc[-1]))
    print("Net Value: " + str(daily_values.iloc[-1] - 200000))
    print("Cumulative Returns: " + str(daily_values.iloc[-1] / daily_values.iloc[0] - 1))
    print("Baseline Cumulative Returns: " + str(baseline_values.iloc[-1] / baseline_values.iloc[0] - 1))
    print("Baseline Net Value: " + str(baseline_values.iloc[-1] - 200000))


def RAW():
    df = get_data(0, 0, freq_per_second=1, directory = "data/", orderbook_filename = "orderbook_1.csv", message_filename = "message_1.csv")
    X = (df['b_size_0'] / df['a_size_0']).to_numpy().reshape(-1, 1)
    Y = (df['price'].shift(-1) / df['price']).to_numpy().reshape(-1, 1)
    Y[-1] = 0

    learner = learners.FFNNLearner(input_size=X[0].shape[0], output_size=Y[0].shape[0], hidden_sizes=[5, 5, 5], learning_rate=0.0001, learning_decay=1)
    learner.train(X, Y, epochs=1000, verbose_freq=1)

    output = learner.test(X)
    output = np.where(output > 1, 1000, -1000)
    trades = pd.Series(output.squeeze(), index=df.index)
    trades.iloc[-1] = 0
    trades.iloc[1:] = trades.diff().iloc[1:]
    daily_values = backtest(df, trades)
    print(f"Net result: {round(daily_values.iloc[-1] - 200000, 2)}")
    # learner.save_model("raw_model.npy")

    # learner.load_model("model.npy")
    # output = learner.test(X)
    # output = np.where(output > 1, 1000, -1000)
    # trades = pd.Series(output.squeeze(), index=df.index)
    # trades.iloc[-1] = 0
    # trades.iloc[1:] = trades.diff().iloc[1:]
    # daily_values = backtest(df, trades)
    # print(f"Net result 2: {round(daily_values.iloc[-1] - 200000, 2)}")

trip_times = []

def QL():
    # 1/60/61
    df = get_data(0, 0, freq_per_second=1/60, directory = "data/", orderbook_filename = "orderbook_1.csv", message_filename = "message_1.csv")
    df = df.iloc[:-1] # size 23k
    print(df.shape)
    learner = learners.LobsterLearner(input_size=2, output_size=3, hidden_sizes=[5, 5, 5, 5], data=df, gamma=0.1, learning_rate=0.99, learning_decay=0.995, floating_cost=0)

    for trip in range(1501):
        start_time = datetime.datetime.now()
        learner.train(df)
        trades = pd.Series(0, index=df.index)
        for i in range(df.shape[0]):
            X = learner.test(i)
            trades.iloc[i] = (X - 1) * 1000
        trades.iloc[-1] = 0
        trades.iloc[1:] = trades.diff().iloc[1:]
        daily_values = backtest(df, trades)
        trip_times.append((datetime.datetime.now() - start_time).total_seconds())
        if trip % 1 == 0:
            print(f"Trip {trip} net result: {round(daily_values.iloc[-1] - 200000, 2)}")
            print("Learning Rate: ", round(learner.learning_rate, 5))
            print("Time Remaining: ", round(np.mean(trip_times) / 60 * (1500 - trip), 2))
            # test1 = nn.test(np.array([[5]]))[0]
            # test2 = nn.test(np.array([[0.3]]))[0]
            # print(test1[2] - test1[0])
            # print(test2[0] - test2[2])

def compare_signs(df1, df2):
    # Apply numpy.sign to get the signs of the elements in the DataFrames
    df1_sign = np.sign(df1)
    df2_sign = np.sign(df2)

    # Compare the signs. The result will be a DataFrame of booleans.
    # True indicates that the signs match, False indicates that they do not.
    sign_match = df1_sign == df2_sign
    sign_match_int = sign_match.astype(int)


    return sign_match_int

def baseline(df, start_value):
    trades = pd.Series(0, index=df.index)
    trades.iloc[0] = 1000
    trades.iloc[-1] = 0
    daily_values = backtest(df, trades, starting_value=start_value)
    return daily_values
     

     

def test_FFNN_daily():
        
        durations = [1/6, 1, 100,1000]
        IS_accs = []
        OOS_accs = []
        oos_accs = []
        X_axis = np.arange(len(durations))
        for duration in durations:
            df = get_data(0, 1000000, freq_per_second=duration, directory = "data/", orderbook_filename = "orderbook_1.csv", message_filename = "message_1.csv")
            X = (df['b_size_0'] / df['a_size_0']).to_numpy().reshape(-1, 1)
            Y = (df['price'].shift(-1) / df['price']).to_numpy().reshape(-1, 1)
            Y[-1] = 0
            if duration == 1/6 or duration == 1: 
                learner = learners.FFNNLearner(input_size=X[0].shape[0], output_size=Y[0].shape[0], hidden_sizes=[5, 5, 5], learning_rate=0.00001, learning_decay=1)
                learner.train(X, Y, epochs = 15, verbose_freq=1)
            else:
                learner = learners.FFNNLearner(input_size=X[0].shape[0], output_size=Y[0].shape[0], hidden_sizes=[5, 5, 5], learning_rate=0.000001, learning_decay=1)
                learner.train(X, Y, epochs = 15, verbose_freq=1)


            is_results = learner.test(X)
         

            preds = Y - 1
            results = preds[preds != 0]
            is_results = is_results[preds!=0] -1
            final = compare_signs(results, is_results)

            IS_acc = final.sum(axis = 0)/final.shape[0]
            IS_accs.append(IS_acc)
            print("IS accuracy: "+ str(IS_acc))


            msgs = sorted(glob.glob('./data/monthly/AAPL_2023-04-*_message_10.csv'))
            orders = sorted(glob.glob('./data/monthly/AAPL_2023-04-*_orderbook_10.csv'))
        
            for x in range(2):
                orderbook_file = orders[x]
                msg_file = msgs[x]
                df = get_data(0, 1000000, freq_per_second=duration, directory = "", orderbook_filename = orderbook_file, message_filename = msg_file)
                X = (df['b_size_0'] / df['a_size_0']).to_numpy().reshape(-1, 1)
                Y = (df['price'].shift(-1) / df['price']).to_numpy().reshape(-1, 1)
                Y[-1] = 0
                
                os_results = learner.test(X)

                preds = Y - 1
                results = preds[preds != 0]
                os_results = os_results[preds!=0] -1
                final = compare_signs(results, os_results)

                OOS_acc = final.sum(axis = 0)/final.shape[0]

                OOS_accs.append(OOS_acc)
            print("Completed: " + str(duration))
            oos_accs.append(sum(OOS_accs)/len(OOS_accs))


            
            



            #plt.plot(is_results-1)
            #plt.plot(preds -1)
        
        plt.bar(X_axis - 0.2, IS_accs, 0.4, label="In Sample accuracy")
        plt.bar(X_axis + 0.2, oos_accs, 0.4, label=" Out of Sample accuracy")
        plt.legend(loc = 'upper left')
        plt.title("Accuracy of FFNN for Predicting Up/Down Price Movements Based on Volume Ratio for Different Frequencies")
        durations = ["%.2f" % duration for duration in durations]
        plt.xticks(X_axis, durations)
        plt.xlabel('Frequency of Data (per second)')
        plt.ylabel('Accurcacy (%)')
        plt.show()

    
def test_FNNN_2day():

    msgs = sorted(glob.glob('./data/monthly/AAPL_2023-04-*_message_10.csv'))
    orders = sorted(glob.glob('./data/monthly/AAPL_2023-04-*_orderbook_10.csv'))
    
    for x in range(1):
        orderbook_file = orders[x]
        msg_file = msgs[x]
        df = get_data(0, 100000, freq_per_second=1, directory = "", orderbook_filename = orderbook_file, message_filename = msg_file)
        df = df.iloc[:-1]
        
        

    pass
         

def test_Qlearner_daily():

    
    durations = [1,100 ,1000]
    IS_accs = []
    OOS_accs = []
    bl_is = []
    bl_oos = []
    X_axis = np.arange(len(durations))
    for duration in durations:
        df = get_data(0, 100000, freq_per_second=duration, directory = "data/", orderbook_filename = "orderbook_1.csv", message_filename = "message_1.csv")
        df = df.iloc[:-1]
        print(df)
        
        split_index = int(df.shape[0] * 0.5)
        oo_sample = df.iloc[split_index:]
        in_sample = df.iloc[:split_index]

        learner = learners.LobsterLearner(input_size=2, output_size=3, hidden_sizes=[5, 5, 5, 5], data=in_sample, gamma=0.9, learning_rate=0.99, learning_decay=0.99)

        for trip in range(10):
            learner.train(in_sample)
            trades = pd.Series(0, index=in_sample.index)
            for i in range(in_sample.shape[0]):
                X = learner.test(i)
                trades.iloc[i] = (X - 1) * 1000
            trades.iloc[-1] = 0
            trades.iloc[1:] = trades.diff().iloc[1:]
            daily_values = backtest(in_sample, trades)
            if trip % 1 == 0:
                print(f"Trip {trip} net result: {round(daily_values.iloc[-1] - 200000, 2)}")
                print("Learning Rate: ", learner.learning_rate)
                # test1 = nn.test(np.array([[5]]))[0]
                # test2 = nn.test(np.array([[0.3]]))[0]
                # print(test1[2] - test1[0])
                # print(test2[0] - test2[2])
        bl = baseline(in_sample)
        bl_is.append(bl)
        IS_accs.append(daily_values)


        trades = pd.Series(0, index=oo_sample.index)
        for i in range(oo_sample.shape[0]):
            X = learner.test(i)
            trades.iloc[i] = (learner.test(X) - 1) * 1000
        trades.iloc[-1] = 0
        trades.iloc[1:] = trades.diff().iloc[1:]
        daily_values = backtest(oo_sample, trades)

        OOS_accs.append(daily_values)
        bl = baseline(oo_sample)
        bl_oos.append(bl)
    
    for x in range(len(IS_accs)):
        plt.plot(IS_accs[x])
        plt.plot(bl_is[x])
        plt.show()

        plt.plot(OOS_accs[x])
        plt.plot(bl_oos[x])
        plt.show()
    




def test_Qlearner_monthly():
    
    msgs = sorted(glob.glob('./data/monthly/AAPL_2023-04-*_message_10.csv'))
    orders = sorted(glob.glob('./data/monthly/AAPL_2023-04-*_orderbook_10.csv'))
    overall_bl = np.array([166590])
    overall_model = np.array([166590])
    idx = 0

    freq = 100
    strfreq = "100"
    learner = learners.LobsterLearner(input_size=2, output_size=3, hidden_sizes=[5, 5, 5, 5], data="", gamma=0.9, learning_rate=0.99, learning_decay=0.99)
    learner.load_model("./models/"+strfreq+"/ql_model_"+strfreq+".npy")
    prev_value = 166590
    bl_prev = 166590
 
    end = 0
    for x in range(len(msgs)):
        orderbook_file = orders[x]
        msg_file = msgs[x]
        df = get_data(0, end, freq_per_second=freq, directory = "", orderbook_filename = orderbook_file, message_filename = msg_file)
        df = df.iloc[:-1]
        learner = learners.LobsterLearner(input_size=2, output_size=3, hidden_sizes=[5, 5, 5, 5], data=df, gamma=0.9, learning_rate=0.99, learning_decay=0.99)
        learner.load_model("./models/"+strfreq+"/ql_model_"+strfreq+".npy")
        
        trades = pd.Series(0, index=df.index)
        for i in range(df.shape[0]):
            X = learner.test(i)
            trades.iloc[i] = (X - 1) * 1000
        trades.iloc[-1] = 0
        trades.iloc[1:] = trades.diff().iloc[1:]
        daily_values = backtest(df, trades,starting_value=prev_value)
        prev_value = daily_values.iloc[-1]

       
        overall_model = np.concatenate((overall_model, daily_values))
      
        bl = baseline(df, bl_prev)
        
        overall_bl = np.concatenate((overall_bl, bl))
        bl_prev = bl.iloc[-1]
        #plt.plot(daily_values)
        #plt.plot(bl)
        plt.plot(daily_values, label ="Model 1000")
        plt.plot(bl, label ="Baseline 1000")
        #plt.show()
    
    print("Model 100  stats")
    print(prev_value)
    print(bl_prev)
    
    freq = 1000
    strfreq = "1000"
    learner = learners.LobsterLearner(input_size=2, output_size=3, hidden_sizes=[5, 5, 5, 5], data="", gamma=0.9, learning_rate=0.99, learning_decay=0.99)
    learner.load_model("./models/"+strfreq+"/ql_model_"+strfreq+".npy")
    prev_value = 166590
    bl_prev = 166590
    overall_bl = np.array([166590])
    overall_model = np.array([166590])

    for x in range(len(msgs)):
        orderbook_file = orders[x]
        msg_file = msgs[x]
        df = get_data(0, end, freq_per_second=freq, directory = "", orderbook_filename = orderbook_file, message_filename = msg_file)
        df = df.iloc[:-1]
        learner = learners.LobsterLearner(input_size=2, output_size=3, hidden_sizes=[5, 5, 5, 5], data=df, gamma=0.9, learning_rate=0.99, learning_decay=0.99)
        learner.load_model("./models/"+strfreq+"/ql_model_"+strfreq+".npy")


        
        trades = pd.Series(0, index=df.index)
        for i in range(df.shape[0]):
            X = learner.test(i)
            trades.iloc[i] = (X - 1) * 1000
        trades.iloc[-1] = 0
        trades.iloc[1:] = trades.diff().iloc[1:]
        daily_values = backtest(df, trades,starting_value=prev_value)
        prev_value = daily_values.iloc[-1]

       
        overall_model = np.concatenate((overall_model, daily_values))
      
        
        bl = baseline(df, bl_prev)
        
        overall_bl = np.concatenate((overall_bl, bl))
        bl_prev = bl.iloc[-1]
            
        plt.plot(daily_values, label ="Model 100")
        #plt.plot(daily_values)
        #plt.plot(bl)
        #plt.show()\


    print("Model 1000  stats")
    print(prev_value)
    print(bl_prev)

    freq = 1
    strfreq = "1"
    learner = learners.LobsterLearner(input_size=2, output_size=3, hidden_sizes=[5, 5, 5, 5], data="", gamma=0.9, learning_rate=0.99, learning_decay=0.99)
    learner.load_model("./models/"+strfreq+"/ql_model_"+strfreq+".npy")
    prev_value = 166590
    bl_prev = 166590
    overall_bl = np.array([166590])
    overall_model = np.array([166590])

    for x in range(len(msgs)):
        orderbook_file = orders[x]
        msg_file = msgs[x]
        df = get_data(0, end, freq_per_second=freq, directory = "", orderbook_filename = orderbook_file, message_filename = msg_file)
     
        df = df.iloc[:-1]
        
        learner = learners.LobsterLearner(input_size=2, output_size=3, hidden_sizes=[5, 5, 5, 5], data=df, gamma=0.9, learning_rate=0.99, learning_decay=0.99)
        learner.load_model("./models/"+strfreq+"/ql_model_"+strfreq+".npy")

 
    
        
        trades = pd.Series(0, index=df.index)
        for i in range(df.shape[0]):
            X = learner.test(i)
            trades.iloc[i] = (X - 1) * 1000
        trades.iloc[-1] = 0
        trades.iloc[1:] = trades.diff().iloc[1:]
        daily_values = backtest(df, trades,starting_value=prev_value)
        prev_value = daily_values.iloc[-1]

       
        overall_model = np.concatenate((overall_model, daily_values))
      
        
        bl = baseline(df, bl_prev)
        
        overall_bl = np.concatenate((overall_bl, bl))
        bl_prev = bl.iloc[-1]
            
        plt.plot(daily_values, label ="Model 1")
        #plt.plot(daily_values)
        #plt.plot(bl)
        #plt.show()
    
    print("Model 1 stats")
    print(prev_value)
    print(bl_prev)
    
    freq = 1/60
    strfreq = "1.60"
    learner = learners.LobsterLearner(input_size=2, output_size=3, hidden_sizes=[5, 5, 5, 5], data="", gamma=0.9, learning_rate=0.99, learning_decay=0.99)
    learner.load_model("./models/"+strfreq+"/ql_model_"+strfreq+".npy")
    prev_value = 166590
    bl_prev = 166590
    overall_bl = np.array([166590])
    overall_model = np.array([166590])

    for x in range(1):
        orderbook_file = orders[x]
        msg_file = msgs[x]
        df = get_data(0, end, freq_per_second=freq, directory = "", orderbook_filename = orderbook_file, message_filename = msg_file)
       
        df = df.iloc[:-1]
        
        learner = learners.LobsterLearner(input_size=2, output_size=3, hidden_sizes=[5, 5, 5, 5], data=df, gamma=0.9, learning_rate=0.99, learning_decay=0.99)
        learner.load_model("./models/"+strfreq+"/ql_model_"+strfreq+".npy")

      
    
        
        trades = pd.Series(0, index=df.index)
        for i in range(df.shape[0]):
            X = learner.test(i)
            trades.iloc[i] = (X - 1) * 1000
        trades.iloc[-1] = 0
        trades.iloc[1:] = trades.diff().iloc[1:]
        daily_values = backtest(df, trades,starting_value=prev_value)
        prev_value = daily_values.iloc[-1]

       
        overall_model = np.concatenate((overall_model, daily_values))
      
        
        bl = baseline(df, bl_prev)
        
        overall_bl = np.concatenate((overall_bl, bl))
        bl_prev = bl.iloc[-1]
            
        plt.plot(daily_values, label ="Model 1/60")
        #plt.plot(daily_values)
        #plt.plot(bl)
        #plt.show()

    print("Model 1/60 stats")
    print(prev_value)
    print(bl_prev)

    plt.title("Full month of trading for both the baseline and the Model for AAPL data at 1/60 frequency")
    plt.legend(loc = "upper left")
    plt.xlabel("")
    plt.ylabel("Porfolio value")
    plt.show()




def testing():
        
        test_FFNN_daily()
        #test_FNNN_2day()
        #QL()
        #test_Qlearner_daily()
        #test_Qlearner_monthly()
       

        

# cProfile.run('run()', sort='tottime')
testing()

import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import datetime

def run_lstm(X, y, test_price, data_portion, layer1 = 50, layer2=30, batch = 100, epoch = 150):
    print(test_price.shape)
    logger = mp.get_logger()  # Get the logger set up for multiprocessing
    logger.debug(f'Initing model')
    #strategy = tf.distribute.MirroredStrategy()

    scaler = MinMaxScaler()
    model = Sequential()



    # find max and min of all series in X

    #scaled_data = scaler.fit_transform(X)
    
    
    # maybe try without transfrom?
    
    #plt.plot(scaled_data[0])

    #plt.show()
    #split into x and y data
    y = X[:,data_portion:]

    X = X[:,:data_portion]
   
    # issues: work wiht scaler
    scaled_test = test_price.reshape(1, -1)

    test_x = scaled_test[0][:data_portion]
    test_y = scaled_test[0][data_portion:]
    test_y = test_y.reshape(1, test_y.shape[0])
    test_x = test_x.reshape(1, X.shape[1], 1)

    model.add(LSTM(units=layer1 ,return_sequences=False, input_shape=(X.shape[1], 1)))

    model.add(Dense(units=layer2, activation='relu'))
    model.add(Dense(units=y.shape[1], activation='linear'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    losses = np.zeros((epoch))
    logger.debug(f'Fitting model')
    for i in range(epoch):
        
        model.fit(X, y, epochs=1,batch_size=batch, verbose=1)
        # get test loss
        test_loss = model.evaluate(test_x, test_y, verbose=0)
        losses[i] = test_loss[0]
        print(f"Epoch {i}/{epoch}")
        

    logger.debug(f'Finished fitting model')
    plt.plot(losses)
    plt.show()
    #test
    predicted_price = model.predict(test_x)

    # combine test_x and predicted price
    test_x = test_x.reshape(test_x.shape[1])
    predicted_price = np.concatenate((test_x, predicted_price[0]), axis=0).reshape(1, -1)
    
    #invert scaling
    #predicted_price = predicted_price

    print("inference time")
    print(f"Predicted Price for Next Day: {predicted_price}")
    print(test_price.shape)
    print(predicted_price.shape)
    plt.plot(test_price)
    plt.plot(predicted_price[0])
    plt.show()

    return(predicted_price[0][-1], test_price[data_portion], test_price[-1])



    """
    Create a new lstm for each cluster and train on each cluster
    """
def cluster_lstm(clusters, test_cluster, test_price, data_portion, layer1 = 50, batch = 100, epoch = 150):
    print(test_price)
    models = []
    for x in range(len(clusters)):


        cluster_series = np.empty((len(clusters[x]),data_portion))
        for y in range(len(clusters[x])):
            cluster_series[y] = clusters[x][y][0:data_portion]

        
        cluster_y = np.empty((len(clusters[x]),data_portion))
        for y in range(len(clusters[x])):
            cluster_y[y] = clusters[x][y][data_portion:]

        print(cluster_series)
        print(cluster_y)

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(cluster_series)
        scaled_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)
        print(scaled_data.shape)
        #scaled_test = scaler.transform(test_price[:data_portion].values.reshape(1, -1))
        #scaled_test = test_price.reshape(1, 11700, 1)
    

        model = Sequential()
        model.add(LSTM(units=layer1,return_sequences=False, input_shape=(scaled_data.shape[1], 1)))
        model.add(Dense(units=cluster_y.shape[1]))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(scaled_data, cluster_y, epochs=epoch, batch_size=batch)

        # if model rmse is too high train 10 more epochs

    ttest = scaler.transform(test_price[0:data_portion].reshape(1, -1))
    # test performance of each model on test series
    test_predictions = []
    for x in range(len(models)):
        test_predictions.append(models[x].predict(ttest.reshape(1, ttest.shape[1], 1)))

    print(test_cluster+1)
    for x in range(len(test_predictions)):
        print(x+1, test_predictions[x], test_price[-1], mean_squared_error([test_price[-1]], test_predictions[x]))
    
    print(test_predictions[test_cluster][0][-1], test_price[-1], test_price[0])
    
    return(test_predictions[test_cluster][0][-1], test_price[-1], test_price[0])



def simple_lstm():

    import pandas as pd

    df = pd.read_csv('/Users/lspieler/Downloads/MSFT (2).csv')

    # combine with aapl data
    df2 = pd.read_csv('/Users/lspieler/Downloads/AAPL.csv')
    df = pd.concat([df,df2])

    df = df.dropna()
    df = df[['Date', 'Close']]
    df['Date'] = df['Date'].apply(str_to_datetime)
    df.index = df.pop('Date')
    plt.plot(df.index, df['Close'])
    plt.show()

    windowed_df = df_to_windowed_df(df, 
                                '2021-03-25', 
                                '2022-03-23', 
                                n=3)
    
    print(windowed_df.head())

    dates, X, y = windowed_df_to_date_X_y(windowed_df)

    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

    plt.plot(dates_train, y_train)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, y_test)

    plt.legend(['Train', 'Validation', 'Test'])
    plt.show()



    model = Sequential([Input((3, 1)),
                        LSTM(100),
                        Dense(32, activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(1)])

    model.compile(loss='mse', 
                optimizer=Adam(learning_rate=0.001),
                metrics=['mean_absolute_error'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

    train_predictions = model.predict(X_train).flatten()

    plt.plot(dates_train, train_predictions)
    plt.plot(dates_train, y_train)
    plt.legend(['Training Predictions', 'Training Observations'])
    plt.show()

    val_predictions = model.predict(X_val).flatten()

    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.legend(['Validation Predictions', 'Validation Observations'])
    plt.show()

    test_predictions = model.predict(X_test).flatten()

    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.legend(['Testing Predictions', 'Testing Observations'])

    plt.plot(dates_train, train_predictions)
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.legend(['Training Predictions', 
                'Training Observations',
                'Validation Predictions', 
                'Validation Observations',
                'Testing Predictions', 
                'Testing Observations'])
    plt.show()
        
        

def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
  first_date = str_to_datetime(first_date_str)
  last_date  = str_to_datetime(last_date_str)

  target_date = first_date
  
  dates = []
  X, Y = [], []

  last_time = False
  while True:
    df_subset = dataframe.loc[:target_date].tail(n+1)
    
    if len(df_subset) != n+1:
      print(f'Error: Window of size {n} is too large for date {target_date}')
      return

    values = df_subset['Close'].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
    
    if last_time:
      break
    
    target_date = next_date

    if target_date == last_date:
      last_time = True
    
  ret_df = pd.DataFrame({})
  ret_df['Target Date'] = dates
  
  X = np.array(X)
  for i in range(0, n):
    X[:, i]
    ret_df[f'Target-{n-i}'] = X[:, i]
  
  ret_df['Target'] = Y

  return ret_df



def windowed_df_to_date_X_y(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

if __name__ == "__main__":
    simple_lstm()
 # run simple linear regression learner on the data 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tqdm import tqdm



def simple_multi_linear(data, target, test_x, test_y, test_size = 0.2, random_state = 0):

    # create the model
    model = LinearRegression()




    model.fit(data, target)

    # make predictions
    predictions = model.predict(test_x)
    
    # calculate the mean squared error
    mse = mean_squared_error(test_y, predictions)

    # calculate the r2 score
    r2 = r2_score(test_y, predictions)

    return mse, r2

   
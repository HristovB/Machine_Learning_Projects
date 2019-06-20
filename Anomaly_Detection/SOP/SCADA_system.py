import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score


# setup data
def data_setup(data):
    # separate datetime column from other data for easier setup
    datetime = data.get(['DATETIME'])
    X = data.drop(['DATETIME'], 1)

    # scale data in range between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(X)
    X.loc[:, :] = scaled_values

    # # standardize data with a Gaussian distribution (0 mean, 1 std) for easier detection of outliers (anomalies)
    # scaler = StandardScaler().fit(X)
    # scaled_values = scaler.transform(X)
    # X.loc[:, :] = scaled_values

    # setup datetime column
    dt_vals = datetime.values
    dt_vals = dt_vals.flatten()
    date, time = zip(*(element.split(' ') for element in dt_vals))
    day, month, year = zip(*(element.split('/') for element in date))

    time = np.asarray(time)
    day = np.asarray(day)
    month = np.asarray(month)
    year = np.asarray(year)

    time = np.asarray([element.lstrip('0') for element in time])
    time[time == ''] = '0'

    day = np.asarray([element.lstrip('0') for element in day])
    month = np.asarray([element.lstrip('0') for element in month])

    time = time.astype(np.int)
    day = day.astype(np.int)
    month = month.astype(np.int)
    year = year.astype(np.int)

    datetime = pd.DataFrame({'Time_Sine': np.transpose(np.sin(time)), 'Time_Cosine': np.transpose(np.cos(time)), 'Day_Sine': np.transpose(np.sin(day)), 'Day_Cosine': np.transpose(np.cos(day)), 'Month_Sine': np.transpose(np.sin(month)), 'Month_Cosine': np.transpose(np.cos(month)), 'Year': np.transpose(year)})

    output = pd.concat([datetime, X], 1)

    return output


# calculate the mean and variance of a dataset
def estimate_gaussian(data):
    u = data.mean()

    m, n = data.shape

    sigma = np.zeros((n, n))

    for i in range(0, m):
        sigma = sigma + (data.iloc[i] - u).values.reshape(n, 1).dot((data.iloc[i] - u).values.reshape(1, n))

    sigma = sigma * (1.0/m)

    return u.values, sigma


# train_data = pd.read_csv('dataset03.csv')
# validation_data = pd.read_csv('dataset04.csv')
# test_data = pd.read_csv('dataset05.csv')

# train_data = data_setup(train_data)
# validation_data = data_setup(validation_data)
# test_data = data_setup(test_data)
#
# train_data.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\train_data.csv', index=False)
# validation_data.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\validation_data.csv', index=False)
# test_data.to_csv(r'C:\Users\Blagoj\PycharmProjects\Machine_Learning\Anomaly_Detection\SOP\test_data.csv', index=False)

train_data = pd.read_csv('train_data.csv')
validation_data = pd.read_csv('validation_data.csv')
test_data = pd.read_csv('test_data.csv')

u, sigma = estimate_gaussian(train_data)

print(u, sigma)


import numpy as np
import pandas as pd
from sklearn.neural_network import multilayer_perceptron
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.externals import joblib

raw_data = pd.read_csv('train.csv')
raw_test = pd.read_csv('test.csv')

test_X = raw_test.values

data_X = raw_data.values[:, 1:]
data_y = raw_data.values[:, 0]

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=157)

# nn = multilayer_perceptron.MLPClassifier(learning_rate_init=0.0003, hidden_layer_sizes=(784, 28),
#                                          activation='logistic', solver='adam', verbose=True)
# nn.fit(X_train, y_train)

# filename = '/home/bhristov/Documents/Programming/PyCharm/Machine_Learning/Kaggle/Digit_Recognizer/neural_net.joblib.pkl'
# _ = joblib.dump(nn, filename, compress=9)

nn = joblib.load('neural_net.joblib.pkl')

y_predicted = nn.predict(X_test)
score = f1_score(y_true=y_test, y_pred=y_predicted, average='micro')
print(score)

# y_output = nn.predict(test_X)
#
# filename = '/home/bhristov/Documents/Programming/PyCharm/Machine_Learning/Kaggle/Digit_Recognizer/output.csv'
# output_id = np.arange(1, len(y_output)+1)
# y_csv = pd.DataFrame({'ImageId': output_id, 'Label': y_output})
# y_csv.set_index('ImageId', inplace=True)
# y_csv.to_csv(filename)

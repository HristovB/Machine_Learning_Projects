import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, LabelBinarizer, Normalizer
from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

training_data = pd.read_csv("F:\Stuff\Data\\training_set.csv")
training_metadata = pd.read_csv("F:\Stuff\Data\\training_set_metadata.csv")

# One-Hot encoding the target classes

data_y = training_metadata.get(['target'])
binarizer = LabelBinarizer()
data_y = pd.DataFrame(binarizer.fit_transform(data_y.values), columns=['6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95'])

##

# data_615 = training_data.loc[training_data['object_id'] == 615]

# plot_data = data_615.get(['mjd', 'flux', 'passband']).sort_values(by=['passband'])

# plot_data = plot_data.loc[plot_data['mjd'] < 62000]

# colors = ["#A025BE", "#25BE2C", "#DF2020", "#E89113", "#254CCF", "#000000"]
# palette = sns.color_palette(colors)
#
# ax = sns.lineplot(x='mjd', y='flux', hue='passband', data=plot_data, palette=palette)
#
# ax.set_title('Light Curve of 615')
# ax.set_xlabel('MJD')
# ax.set_ylabel('Flux')
#
# plt.show()
##



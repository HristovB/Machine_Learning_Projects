import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
from Kaggle.Common_Files.utils import set_seed, train_network, correlation_plot

matplotlib.use('TkAgg')
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def clean_data(dataset, data_type='train'):
    dataset.loc[dataset['Alley'].isna(), 'Alley'] = 'NA'
    dataset.loc[dataset['LotFrontage'].isna(), 'LotFrontage'] = 0.0
    dataset.loc[dataset['MasVnrArea'].isna(), 'MasVnrArea'] = 0.0
    dataset.loc[dataset['BsmtFinSF1'].isna(), 'BsmtFinSF1'] = 0.0
    dataset.loc[dataset['BsmtFinSF2'].isna(), 'BsmtFinSF2'] = 0.0
    dataset.loc[dataset['BsmtUnfSF'].isna(), 'BsmtUnfSF'] = 0.0
    dataset.loc[dataset['TotalBsmtSF'].isna(), 'TotalBsmtSF'] = 0.0
    dataset.loc[dataset['BsmtFullBath'].isna(), 'BsmtFullBath'] = 0.0
    dataset.loc[dataset['BsmtHalfBath'].isna(), 'BsmtHalfBath'] = 0.0
    dataset.loc[dataset['GarageCars'].isna(), 'GarageCars'] = 0.0
    dataset.loc[dataset['GarageArea'].isna(), 'GarageArea'] = 0.0
    dataset.loc[dataset['BsmtQual'].isna(), 'BsmtQual'] = 'NA'
    dataset.loc[dataset['BsmtCond'].isna(), 'BsmtCond'] = 'NA'
    dataset.loc[dataset['BsmtExposure'].isna(), 'BsmtExposure'] = 'NA'
    dataset.loc[dataset['BsmtFinType1'].isna(), 'BsmtFinType1'] = 'NA'
    dataset.loc[dataset['BsmtFinType2'].isna(), 'BsmtFinType2'] = 'NA'
    dataset.loc[dataset['FireplaceQu'].isna(), 'FireplaceQu'] = 'NA'
    dataset.loc[dataset['GarageType'].isna(), 'GarageType'] = 'NA'
    dataset.loc[dataset['GarageYrBlt'].isna(), 'GarageYrBlt'] = 0
    dataset.loc[dataset['GarageFinish'].isna(), 'GarageFinish'] = 'NA'
    dataset.loc[dataset['GarageQual'].isna(), 'GarageQual'] = 'NA'
    dataset.loc[dataset['GarageCond'].isna(), 'GarageCond'] = 'NA'
    dataset.loc[dataset['PoolQC'].isna(), 'PoolQC'] = 'NA'
    dataset.loc[dataset['Fence'].isna(), 'Fence'] = 'NA'
    dataset.loc[dataset['MiscFeature'].isna(), 'MiscFeature'] = 'NA'

    if data_type == 'train':
        dataset = dataset[dataset['MasVnrType'].notna()]
        dataset = dataset[dataset['Electrical'].notna()]
        dataset = dataset[dataset['MSZoning'].notna()]
        dataset = dataset[dataset['Exterior1st'].notna()]
        dataset = dataset[dataset['SaleType'].notna()]

    elif data_type == 'test':
        dataset.loc[dataset['MasVnrType'].isna(), 'MasVnrType'] = dataset['MasVnrType'].value_counts().index[0]
        dataset.loc[dataset['Electrical'].isna(), 'Electrical'] = dataset['Electrical'].value_counts().index[0]
        dataset.loc[dataset['MSZoning'].isna(), 'MSZoning'] = dataset['MSZoning'].value_counts().index[0]
        dataset.loc[dataset['Exterior1st'].isna(), 'Exterior1st'] = dataset['Exterior1st'].value_counts().index[0]
        dataset.loc[dataset['Exterior2nd'].isna(), 'Exterior2nd'] = dataset['Exterior2nd'].value_counts().index[0]
        dataset.loc[dataset['KitchenQual'].isna(), 'KitchenQual'] = dataset['KitchenQual'].value_counts().index[0]
        dataset.loc[dataset['Functional'].isna(), 'Functional'] = dataset['Functional'].value_counts().index[0]
        dataset.loc[dataset['Utilities'].isna(), 'Utilities'] = dataset['Utilities'].value_counts().index[0]
        dataset.loc[dataset['SaleType'].isna(), 'SaleType'] = dataset['SaleType'].value_counts().index[0]

    return dataset


def encode_values(dataset):
    column_names = dataset.loc[:, (dataset.dtypes == 'object').values].columns.values
    encoder = LabelEncoder()

    for column in column_names:
        dataset[column] = encoder.fit_transform(dataset[column])

    return dataset


# noinspection DuplicatedCode
def neural_network():
    from keras.models import Sequential
    from keras.layers import BatchNormalization, Activation, Dense, Dropout
    from keras.initializers import he_uniform
    from keras.regularizers import l2
    from keras.activations import relu
    from keras.constraints import maxnorm

    init_relu = he_uniform(seed=157)    # used for tanh and softmax

    model = Sequential()

    # input and first Dense layer
    model.add(Dense(units=35, kernel_initializer=init_relu, kernel_regularizer=l2(0.01), kernel_constraint=maxnorm(3)))
    model.add(Activation(relu))

    # second Dense layer
    model.add(Dense(units=140, kernel_initializer=init_relu, kernel_regularizer=l2(0.01), kernel_constraint=maxnorm(3)))
    model.add(Activation(relu))

    # fourth Dense layer
    model.add(Dense(units=140, kernel_initializer=init_relu, kernel_regularizer=l2(0.01), kernel_constraint=maxnorm(3)))
    model.add(Activation(relu))

    # fourth Dense layer
    model.add(Dense(units=70, kernel_initializer=init_relu, kernel_regularizer=l2(0.01), kernel_constraint=maxnorm(3)))
    model.add(Activation(relu))

    # fourth Dense layer
    model.add(Dense(units=35, kernel_initializer=init_relu, kernel_regularizer=l2(0.01), kernel_constraint=maxnorm(3)))
    model.add(Activation(relu))

    # output Dense layer
    model.add(Dense(units=1, kernel_initializer=init_relu))
    model.add(Activation(relu))

    return model


if __name__ == '__main__':
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    test_id = test_data['Id']

    train_data = clean_data(train_data)
    test_data = clean_data(test_data, data_type='test')

    train_data = train_data.reset_index(drop=True)
    y_data = train_data['SalePrice']
    X_data = train_data.drop(['Id', 'SalePrice'], axis=1)
    X_test = test_data.drop('Id', axis=1)

    X_data = encode_values(X_data)
    X_test = encode_values(X_test)

    corr_reg = mutual_info_regression(X=X_data, y=y_data, random_state=157)

    X_data = X_data.iloc[:, corr_reg > 0.1]
    X_test = X_test.iloc[:, corr_reg > 0.1]

    # correlation_plot(X_data, y_data, method='label', problem_type='reg')

    scaler = StandardScaler()
    X_data = pd.DataFrame(scaler.fit_transform(X_data), columns=X_data.columns)
    X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_data.columns)

    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import r2_score

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=157)

    # clf = XGBRegressor(n_estimators=2000, tree_method='exact', learning_rate=0.01, colsample_bynode=0.5, colsample_bytree=0.5, colsample_bylevel=0.5, booster='gbtree', reg_lambda=0.01, verbosity=1, random_state=157, n_jobs=-1)
    #
    # clf.fit(X_train, y_train)
    #
    # y_pred = clf.predict(X_val)
    #
    # print(mse(y_val, y_pred, squared=False))
    # print(r2_score(y_val, y_pred))

    import tensorflow as tf
    train_network(train_data=X_train, train_classes=y_train, val_data=X_val, val_classes=y_val, clf=neural_network(), seed=157, learning_rate=0.0001, epochs=5000, batch_size=8,
                  loss_function='mse', metric=tf.keras.metrics.RootMeanSquaredError(name='rmse'), early_stopping=False)

    #
    #
    # test_prediction = pd.DataFrame(clf.predict(X_test), columns=['SalePrice'])
    #
    # submission = pd.concat([test_id, test_prediction], axis=1)
    #
    # submission.to_csv('kaggle_submission_xgboost.csv', index=False)

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.models import load_model

matplotlib.use('TkAgg')
pd.set_option('mode.chained_assignment', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.expand_frame_repr', False)


def set_seed(seed_value):

    import os
    import random
    import tensorflow as tf

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from tensorflow.python.keras import backend as k

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                            inter_op_parallelism_threads=1)

    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    k.set_session(sess)


def reshape_data(dataset, file_name='data', label=False):
    if label:
        data_labels = dataset['label']
        np.save(file_name + '_labels.npy', data_labels)

        dataset = dataset.drop('label', axis=1)

    dataset = dataset.values[:, :, np.newaxis].reshape(-1, 28, 28)
    np.save(file_name + '.npy', dataset)


def visualize_data(dataset, data_labels, seed):
    np.random.seed(seed)
    idxs = np.random.randint(0, dataset.shape[0], 9)

    fig, ax = plt.subplots(3, 3, figsize=(8, 8))

    ax[0, 0].imshow(dataset[idxs[0], :, :], cmap='Greys')
    ax[0, 0].set_title('Handwritten number ' + str(data_labels[idxs[0]]))

    ax[0, 1].imshow(dataset[idxs[1], :, :], cmap='Greys')
    ax[0, 1].set_title('Handwritten number ' + str(data_labels[idxs[1]]))

    ax[0, 2].imshow(dataset[idxs[2], :, :], cmap='Greys')
    ax[0, 2].set_title('Handwritten number ' + str(data_labels[idxs[2]]))

    ax[1, 0].imshow(dataset[idxs[3], :, :], cmap='Greys')
    ax[1, 0].set_title('Handwritten number ' + str(data_labels[idxs[3]]))

    ax[1, 1].imshow(dataset[idxs[4], :, :], cmap='Greys')
    ax[1, 1].set_title('Handwritten number ' + str(data_labels[idxs[4]]))

    ax[1, 2].imshow(dataset[idxs[5], :, :], cmap='Greys')
    ax[1, 2].set_title('Handwritten number ' + str(data_labels[idxs[5]]))

    ax[2, 0].imshow(dataset[idxs[6], :, :], cmap='Greys')
    ax[2, 0].set_title('Handwritten number ' + str(data_labels[idxs[6]]))

    ax[2, 1].imshow(dataset[idxs[7], :, :], cmap='Greys')
    ax[2, 1].set_title('Handwritten number ' + str(data_labels[idxs[7]]))

    ax[2, 2].imshow(dataset[idxs[8], :, :], cmap='Greys')
    ax[2, 2].set_title('Handwritten number ' + str(data_labels[idxs[8]]))

    plt.tight_layout()
    plt.show()


# noinspection DuplicatedCode
def neural_network():
    from keras.models import Sequential
    from keras.layers import BatchNormalization, Activation, Dropout, Dense, MaxPooling2D, Conv2D, Flatten
    from keras.initializers import he_uniform, glorot_uniform
    from keras.activations import relu, tanh
    from keras.constraints import maxnorm
    from keras.regularizers import l2

    init_relu = he_uniform(seed=157)        # used for ReLU
    init_tanh = glorot_uniform(seed=157)    # used for tanh and softmax

    model = Sequential()

    # input and first Convolutional layer

    model.add(Conv2D(name='Conv_1', input_shape=(28, 28, 1), filters=64, kernel_size=(3, 3), padding='valid', kernel_initializer=init_relu, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(MaxPooling2D(name='Pooling_1', pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(name='Conv_2', filters=64, kernel_size=(3, 3), padding='valid', kernel_initializer=init_relu, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(MaxPooling2D(name='Pooling_2', pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(name='FC_1', units=64, kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(tanh))

    # softmax output Dense layer
    model.add(Dense(name='output', units=10, activation='softmax', kernel_initializer=init_tanh, kernel_constraint=maxnorm(3)))

    return model


def train_network(train_data, train_classes, val_data, val_classes, seed, learning_rate, early_stopping=False):
    from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
    from keras.optimizers import Adam
    from keras.models import save_model
    from keras.metrics import BinaryCrossentropy

    set_seed(seed_value=seed)

    opt = Adam(learning_rate=learning_rate)

    clf = neural_network()

    clf.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    checkpoint = ModelCheckpoint('neural_network_checkpoint_training.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    if early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    else:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=1, mode='auto')

    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    clf.fit(x=train_data, y=train_classes,
            epochs=1000,
            validation_data=(val_data, val_classes),
            batch_size=64,
            shuffle=True,
            callbacks=[tensorboard, checkpoint, early_stopping],
            verbose=1)

    save_model(checkpoint.model, 'neural_network_latest_saved.h5')


if __name__ == '__main__':
    # train_data = np.load('train_data.npy')[:, :, :, np.newaxis]
    # train_labels = np.load('train_data_labels.npy')
    #
    # encoder = LabelBinarizer()
    # train_labels = encoder.fit_transform(train_labels)
    #
    # X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.20, random_state=157, stratify=train_labels)
    #
    # train_network(train_data=X_train, train_classes=y_train, val_data=X_val, val_classes=y_val, seed=157, learning_rate=0.001, early_stopping=False)

    test_data = np.load('test_data.npy')[:, :, :, np.newaxis]

    clf = load_model('neural_network_checkpoint_training.h5')

    prediction = clf.predict(test_data)

    prediction = pd.DataFrame(np.argmax(prediction, axis=-1), columns=['Label'])
    img_idx = pd.DataFrame(np.arange(1, len(prediction) + 1), columns=['ImageId'])

    prediction = pd.concat([img_idx, prediction], axis=1)

    prediction.to_csv('kaggle_submission_cnn.csv', index=False)

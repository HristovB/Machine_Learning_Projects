import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


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


def correlation_plot(X, y, method='feature', problem_type='clf'):

    # ----- Correlation plot ----- #

    if method == 'feature':

        # Plot feature correlation
        corr = X.corr()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)

        ticks = np.arange(0, len(X.columns), 1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)

        ax.set_yticks(ticks)
        ax.set_xticklabels(X.columns)
        ax.set_yticklabels(X.columns)

        plt.savefig('feature_correlation.png')
        plt.show()

    elif method == 'label':

        # Plot label correlation

        if problem_type == 'clf':
            from sklearn.feature_selection import mutual_info_classif
            data_mutual_info = mutual_info_classif(X=X, y=y, random_state=157)

        elif problem_type == 'reg':
            from sklearn.feature_selection import mutual_info_regression
            data_mutual_info = mutual_info_regression(X=X, y=y, n_neighbors=3, random_state=157)

        else:
            raise ValueError('Wrong input for problem type! Acceptable inputs: \'clf\', \'reg\'')

        fig, ax = plt.subplots(1, figsize=(12, 8))
        sns.heatmap(data_mutual_info[:, np.newaxis].T, cmap='Blues', cbar=False, linewidths=1, annot=True, ax=ax)
        ax.set_yticks([], [])
        ticks = np.arange(0.5, len(X.columns), 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(X.columns[0:], rotation=45, ha='right', fontsize=8)
        ax.set_title('Feature correlation with output label', fontsize=14, fontweight='bold', y=1.05)
        # plt.subplots_adjust(wspace=0.2)
        plt.tight_layout()
        plt.savefig('output_correlation.png')
        plt.show()

    elif method != 'feature' and method != 'label':
        raise ValueError('Wrong input for method! Acceptable inputs: \'feature\', \'label\'')


def train_network(train_data, train_classes, val_data, val_classes, clf, seed, learning_rate, epochs=100, batch_size=64, loss_function='mse', metric='accuracy', early_stopping=False):
    from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
    from keras.optimizers import Adam
    from keras.models import save_model

    set_seed(seed_value=seed)

    opt = Adam(learning_rate=learning_rate, amsgrad=True)

    clf.compile(optimizer=opt,
                loss=loss_function,
                metrics=[metric])

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
            epochs=epochs,
            validation_data=(val_data, val_classes),
            batch_size=batch_size,
            shuffle=True,
            callbacks=[tensorboard, checkpoint, early_stopping],
            verbose=1)

    save_model(checkpoint.model, 'neural_network_latest_saved.h5')

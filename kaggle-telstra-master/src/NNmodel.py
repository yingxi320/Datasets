import numpy as np

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, Reshape
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import WeightRegularizer as WR
from keras.callbacks import EarlyStopping

class NN_with_CategoryEmbedding(object):

    def __init__(self, modelparams):
        self.model = self.__build_keras_model(modelparams)


    def __build_keras_model(self, params):
        models = []
        for key, value in params.items():
            if key.endswith("_size"):
                params[key] = int(value)
        model_location = Sequential()
        model_location.add(Embedding(1126, params['location_embedding_size'], input_length=1,
                                     W_regularizer = WR(l1 = params['location_embedding_l1'],
                                                        l2 = params['location_embedding_l2'])))
        model_location.add(Reshape(target_shape=(params['location_embedding_size'],)))
        models.append(model_location)

        model_events = Sequential()
        model_events.add(Dense(params['events_model_size'], input_dim = 53,
                               W_regularizer = WR(l1 = params['events_model_l1'],
                                                  l2 = params['events_model_l2'])))
        models.append(model_events)

        model_log = Sequential()
        model_log.add(Dense(params['log_model_size'], input_dim = 321,
                            W_regularizer = WR(l1 = params['log_model_l1'],
                                               l2 = params['log_model_l2'])))
        models.append(model_log)

        model_res = Sequential()
        model_res.add(Dense(params['resource_model_size'], input_dim = 10,
                            W_regularizer = WR(l1 = params['resource_model_l1'],
                                               l2 = params['resource_model_l2'])))
        models.append(model_res)

        model_sev = Sequential()
        model_sev.add(Embedding(5, params['severity_embedding_size'], input_length=1,
                                     W_regularizer = WR(l1 = params['severity_embedding_l1'],
                                                        l2 = params['severity_embedding_l2'])))
        model_sev.add(Reshape(target_shape=(params['severity_embedding_size'],)))
        models.append(model_sev)

        model_rest = Sequential()
        model_rest.add(Dense(params['rest_model_size'], input_dim = 25,
                            W_regularizer = WR(l1 = params['rest_model_l1'],
                                               l2 = params['rest_model_l2'])))
        models.append(model_rest)

        model = Sequential()
        model.add(Merge(models, mode='concat'))
        model.add(Dense(params['dense_1_size'], init='uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(params["dropout_1"]))
        model.add(Dense(params['dense_2_size'], init='uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(params['dropout_2']))
        # A third dense layer?
        if params['dense_3_size'] > 0:
            model.add(Dense(params['dense_3_size'], init='uniform'))
            model.add(Activation('relu'))
            model.add(Dropout(params['dropout_3']))
        model.add(Dense(3, init='uniform'))
        model.add(Activation('softmax'))

        opt = Adam()
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def split_features(self, x):
        features = x.columns
        X_list = []
        # locations
        X_list.append(x[['location']].values.astype(np.int32)-1)
        # events
        event_features = [f for f in features if f[:6]=="event_"]
        X_list.append(x[event_features].values)
        # logfeatures
        log_features = [f for f in features if f[:11]=="logfeatvol_"]
        X_list.append(x[log_features].values)
        # resources
        res_features = [f for f in features if f[:8]=="restype_"]
        X_list.append(x[res_features].values)
        # severity types
        X_list.append(x[['sevtype']].values.astype(np.int32)-1)
        # other features
        # other_features = [f for f in features if f not in ('location','sevtype')]
        X_list.append(x[['num', 'numsh', 'numsh0','numsh1','loc_count', 'nevents',
          'logvolume_count', 'volsumlog',
          'logvolume_min', 'logvolume_mean', 'logvolume_max', 'logvolume_std',
          'logvolume_sum',
        'logvolume_sum_ma9_diff','volsumlog_ma9_diff', 'logfeatvol_203_ma9_diff',
        'lastknown', 'nextknown',
        'ewma02','ewma12','ewma22',
          'nresources','loc_prob_0','loc_prob_1','loc_prob_2']].values)
        # X_list.append(x[other_features].values)
        return X_list

    def prepare_data(self, X, y):
        XX = self.split_features(X)
        Y = np_utils.to_categorical(y, 3)
        return XX, Y

    def fit_val(self, Xtr, ytr, Xval, yval, nb_epoch = 100, early_stopping_rounds = 0):
        params = {"nb_epoch":nb_epoch,
                  "callbacks":[],
                  "show_accuracy":True,
                  "verbose":0,}
        if early_stopping_rounds > 0:
            early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_rounds)
            params['callbacks'].append(early_stopping)

        Xtrain, Ytrain = self.prepare_data(Xtr,ytr)
        Xvalid, Yvalid = self.prepare_data(Xval,yval)

        hist = self.model.fit(Xtrain, Ytrain, validation_data=(Xvalid, Yvalid), **params)
        return hist.history

    def fit(self, Xtr, ytr, nb_epoch = 100):
        params = {"nb_epoch":nb_epoch,
                  "show_accuracy":True,
                  "verbose":0,}
        Xtrain, Ytrain = self.prepare_data(Xtr,ytr)

        hist = self.model.fit(Xtrain, Ytrain, **params)
        return hist.history

    def predict_proba(self, Xtest):
        return self.model.predict_proba(self.split_features(Xtest),verbose = 0)

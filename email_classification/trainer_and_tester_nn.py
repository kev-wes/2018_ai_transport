# Needed project interpreters:
from api import read_data
import os
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import gc
from keras import models
from keras import layers
import scipy
from sklearn.preprocessing import LabelEncoder
from keras import regularizers

# Git
import numpy as np
import math

import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler
# Eof Git

if __name__ == "__main__":
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        prot_file = os.path.join(path, 'protocol.csv')

        # read data and test files
        data = read_data('email_data.csv')
        # Drop rows which are not specified language
        language = input('Enter the language you want to train: ')
        data = data[data.lang.str.contains(language) == True]
        # replace labels with other when smaller than 2% of data
        data['issueTypeCode'].fillna('NA', inplace=True)
        data_count = data.groupby('issueTypeCode')['created'].count()
        data_dict = data_count.to_dict()
        data['issueTypeCode'] = data['issueTypeCode'].apply(lambda x: 'other' \
                if data_dict.get(x) < data.shape[0] / 100 else x)

        # First store columns which must not be one hot encoded
        numeric_cols = ['neg_subj', 'neu_subj', 'posi_subj', 'compound_subj', 'neg_body', 'neu_body',
                                     'posi_body', 'compound_body']
        for col in numeric_cols:
            data[col].fillna(0, inplace=True)
        x_num_data = scipy.sparse.csr_matrix(data[numeric_cols].values)

        # Store label in data frame
        y_data = data[['issueTypeCode']].astype(str).copy()
        le = LabelEncoder()
        le.fit(y_data)
        y_data = le.transform(y_data)
        # Store ID in data frame
        id_data = data[['issueId']].copy()
        y_length = len(set(y_data))
        # Categorical
        categorical_cols = ['assignedGroup', 'created_d', 'created_day', 'created_month',
                            'created_year', 'created_weekday', 'projectKey', 'fromEmail', 'ccEmail',
                            'toEmail']
        cat_data = data[categorical_cols].astype(str).copy()

        cat_data.fillna('NA', inplace=True)

        x_cat_data = cat_data.to_dict(orient='records')
        cat_data = None
        gc.collect()

        # vectorize
        vectorizer = DictVectorizer(sparse=True)
        vec_x_cat_data = vectorizer.fit_transform(x_cat_data)
        x_cat_data = None
        gc.collect()

        # vectorize tfidf fields
        tfidf_vectorizer = TfidfVectorizer(norm='l2',
                                     ngram_range=(1, 2))
        data['txt_main'].fillna('NA', inplace=True)
        data['txt_subj'].fillna('NA', inplace=True)
        data['pos_main'].fillna('NA', inplace=True)
        data['pos_subj'].fillna('NA', inplace=True)
        data['response_txt'].fillna('NA', inplace=True)
        data['sign'].fillna('NA', inplace=True)
        data_txt_main_d = tfidf_vectorizer.fit_transform(data['txt_main'])
        data_txt_subj_d = tfidf_vectorizer.fit_transform(data['txt_subj'])
        data_pos_main_d = tfidf_vectorizer.fit_transform(data['pos_main'])
        data_pos_subj_d = tfidf_vectorizer.fit_transform(data['pos_subj'])
        data_response_d = tfidf_vectorizer.fit_transform(data['response_txt'])
        data_signature_d = tfidf_vectorizer.fit_transform(data['sign'])
        data = None
        gc.collect()

        data_text = hstack([data_txt_main_d, data_txt_subj_d, data_pos_main_d, data_pos_subj_d,
                            data_response_d, data_signature_d])

        data_txt_main_d, data_txt_subj_d, data_pos_main_d, data_pos_subj_d, data_response_d, data_signature_d \
                = None, None, None, None, None, None
        gc.collect()

        # Join categorical and numerical features
        x_data = hstack((data_text, x_num_data, vec_x_cat_data))
        data_text, x_num_data, vec_x_cat_data = None, None, None
        gc.collect()
        # build train & test set
        x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(x_data, y_data, id_data, test_size=0.2, random_state=42)
        x_data, y_data, id_data = None, None, None
        gc.collect()
        # build train & validation
        x_train, x_val, y_train, y_val, id_train, id_val = train_test_split(x_train, y_train, id_train, test_size=0.2, random_state=42)

        # building and assessing a sequential model
        model = models.Sequential()
        model.add(layers.Dense(128, input_shape=(x_train.shape[1],), kernel_regularizer=regularizers.l2(0.01),
                               activity_regularizer=regularizers.l1(0.01)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(y_length, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        classifier = model.fit(x_train,
                               y_train,
                               epochs=20,
                               batch_size=32,
                               validation_data=(x_val, y_val)
                               )
        # Run predictor
        print("Test:")
        print(model.evaluate(x_test, y_test))
        print(classifier.history['acc'])


from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas
import numpy
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

def read_data(dir):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    # Path of the source file (which is the dest file from the hashing function
    src_file = os.path.join(path, dir)
    df = pandas.read_csv(src_file, sep=';', encoding = 'ISO-8859-1')

    # Extend created date columns
    df['created_d'] = pandas.to_datetime(df['created'], unit="ms")
    df['created_day'] = (df['created_d'].dt.day)
    df['created_month'] = (df['created_d'].dt.month)
    df['created_year'] = (df['created_d'].dt.year)
    df['created_weekday'] = (df['created_d'].dt.weekday)

    # df['updated_d'] = pandas.to_datetime(df['updated'], unit="ms")
    # df['updated_day'] = (df['updated_d'].dt.day)
    # df['updated_month'] = (df['updated_d'].dt.month)
    # df['updated_year'] = (df['updated_d'].dt.year)
    # df['updated_weekday'] = (df['updated_d'].dt.weekday)
    #
    # df['lStT_d'] = pandas.to_datetime(df['lastStatusTransistion'], unit="ms")
    # df['lStT_day'] = (df['lStT_d'].dt.day)
    # df['lStT_month'] = (df['lStT_d'].dt.month)
    # df['lStT_year'] = (df['lStT_d'].dt.year)
    # df['lStT_weekday'] = (df['lStT_d'].dt.weekday)

    return df

def tfidf_extractor(text, ngram_range):

    vectorizer = TfidfVectorizer(norm='l2',
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(text.values.astype('U'))
    return features

def one_hot(df):
    for col in df.columns.values:
        if col == 'created_d' or 'created_day' or 'created_month' or 'created_year' or 'created_weekday' or \
            'updated_d' or 'updated_day' or 'updated_month' or 'updated_year' or 'updated_weekday' or \
            'lStT_d' or 'lStT_day' or 'lStT_month' or 'lStT_year' or 'lStT_weekday':
            one_hot = pandas.get_dummies(df[col])
        else:
            one_hot = pandas.get_dummies(df[col], dummy_na=True)
        df = pandas.concat([df, one_hot], axis=1)
        df = df.drop([col], axis=1)
    df = df.replace(to_replace=numpy.nan, value="-1")
    return df

def split(dir, test_size):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    # Path of the source file (which is the dest file from the hashing function
    src_file = os.path.join(path, dir)
    test_dir = os.path.join(path, 'email_test.csv')
    train_dir= os.path.join(path, 'email_train.csv')
    df = pandas.read_csv(src_file, sep=';')
    train, test = train_test_split(df, test_size = test_size, random_state = 42)
    train.to_csv(path_or_buf=train_dir, sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)
    test.to_csv(path_or_buf=test_dir, sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)





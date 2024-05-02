# Needed project interpreters:
# sklearn
import pickle
from sklearn.svm import SVC
from api import tfidf_extractor, read_data, split
from sklearn import metrics
import numpy
import os, csv
import pandas
from pandas import factorize
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import gc
import scipy
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

if __name__ == "__main__":
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        prot_file = os.path.join(path, 'protocol.csv')

        # read data and test files
        data = read_data('email_data.csv')
        # Drop rows where txt body is empty
        # data.dropna(subset=['txt_main'], inplace=True)

        # replace labels with other when smaller than 2% of data
        data['issueTypeCode'].fillna('NA', inplace=True)
        data_count = data.groupby('issueTypeCode')['created'].count()
        data_dict = data_count.to_dict()
        data['issueTypeCode'] = data['issueTypeCode'].apply(lambda x : 'other' \
            if data_dict.get(x) < data.shape[0] / 50 else x)

        # First store columns which must not be one hot encoded
        numeric_cols = ['neg_subj', 'neu_subj', 'posi_subj', 'compound_subj', 'neg_body', 'neu_body',
                                     'posi_body', 'compound_body']
        for col in numeric_cols:
            data[col].fillna(0, inplace=True)
        x_num_data = scipy.sparse.csr_matrix(data[numeric_cols].values)

        # Store label in data frame
        y_data = data[['issueTypeCode']].copy()

        # Store ID in data frame
        id_data = data[['ID']].copy()
        # USE THIS AFTER ADDING THE JIRA ID TO DATASOURCE id_data = data[['issueId']].copy()
        # Categorical
        # categorical_cols = ['assignedGroup', 'assigneeUsername', 'created_d', 'created_day', 'created_month',
        #                      'created_year', 'created_weekday', 'lStT_d', 'lStT_day', 'lStT_month', 'lStT_year',
        #                      'lStT_weekday', 'priorityName', 'projectKey', 'reporterUsername', 'responsibleParty',
        #                      'updated_d', 'updated_day', 'updated_month', 'updated_year', 'updated_weekday']
        categorical_cols = ['assignedGroup', 'created_d', 'created_day', 'created_month',
                            'created_year', 'created_weekday', 'projectKey']
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
        data_txt_main_d = tfidf_vectorizer.fit_transform(data['txt_main'])
        data_txt_subj_d = tfidf_vectorizer.fit_transform(data['txt_subj'])
        data_pos_main_d = tfidf_vectorizer.fit_transform(data['pos_main'])
        data_pos_subj_d = tfidf_vectorizer.fit_transform(data['pos_subj'])
        data = None
        gc.collect()

        data_text = hstack([data_txt_main_d, data_txt_subj_d, data_pos_main_d, data_pos_subj_d])
        data_txt_main_d, data_txt_subj_d, data_pos_main_d, data_pos_subj_d = None, None, None, None
        gc.collect()

        # Join categorical and numerical features
        x_data = hstack((data_text, x_num_data, vec_x_cat_data))
        data_text, x_num_data, vec_x_cat_data = None, None, None
        gc.collect()
        x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(x_data, y_data, id_data, test_size=0.33, random_state=42)
        x_data, y_data, id_data = None, None, None
        gc.collect()

        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100]}]

        scores = ['precision']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                               scoring='%s_macro' % score, verbose=5, n_jobs=-1)
            clf.fit(x_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(x_test)
            print(classification_report(y_true, y_pred))
            print()
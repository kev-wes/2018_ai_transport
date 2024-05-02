# Needed project interpreters:
# sklearn
from api import read_data
import os
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import gc
import scipy
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
        train_it = True
        test_it = True
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        prot_file = os.path.join(path, 'protocol.csv')

        # read data and test files
        data = read_data('email_data.csv')
        # Drop rows where txt body is empty
        # data.dropna(subset=['txt_main'], inplace=True)

        # Drop rows which are not the specified language
        language = input('Enter the language you want to train: ')
        data = data[data.lang.str.contains(language) == True]
        # replace labels with other when smaller than n% of data
        data['issueTypeCode'].fillna('NA', inplace=True)
        data_count = data.groupby('issueTypeCode')['created'].count()
        data_dict = data_count.to_dict()
        data['issueTypeCode'] = data['issueTypeCode'].apply(lambda x : 'other' \
             if data_dict.get(x) < data.shape[0] / 100 else x)
        # Store label in data frame
        y_data = data[['issueTypeCode']].copy()
        # Store ID in data frame
        id_data = data[['issueId']].copy()
        # id_data = data[['ID']].copy()

        # First store columns which must not be one hot encoded
        numeric_cols = ['neg_subj', 'neu_subj', 'posi_subj', 'compound_subj', 'neg_body', 'neu_body',
                                     'posi_body', 'compound_body']
        for col in numeric_cols:
            data[col].fillna(0, inplace=True)
        x_num_data = scipy.sparse.csr_matrix(data[numeric_cols].values)




        # Categorical
        # categorical_cols = ['assignedGroup', 'assigneeUsername', 'created_d', 'created_day', 'created_month',
        #                      'created_year', 'created_weekday', 'lStT_d', 'lStT_day', 'lStT_month', 'lStT_year',
        #                      'lStT_weekday', 'priorityName', 'projectKey', 'reporterUsername', 'responsibleParty',
        #                      'updated_d', 'updated_day', 'updated_month', 'updated_year', 'updated_weekday']
        categorical_cols = ['assignedGroup', 'created_d', 'created_day', 'created_month',
                            'created_year', 'created_weekday', 'projectKey', 'fromEmail', 'ccEmail',
                            'toEmail']
        cat_data = data[categorical_cols].astype(str).copy()
        cat_data.fillna('NA', inplace=True)
        x_cat_data = cat_data.to_dict(orient='records')


        # Empty variable
        cat_data = None
        # Free memory
        gc.collect()

        # vectorize
        vectorizer = DictVectorizer(sparse=True)
        vec_x_cat_data = vectorizer.fit_transform(x_cat_data)
        x_cat_data = None
        gc.collect()

        # vectorize tfidf fields
        tfidf_vectorizer = TfidfVectorizer(norm='l2', ngram_range=(1, 2))
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
        x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(x_data, y_data, id_data, test_size=0.33, random_state=42)
        x_data, y_data, id_data = None, None, None
        gc.collect()
        clf_rf = RandomForestClassifier(n_estimators=20, bootstrap=False, criterion='gini', max_depth=None,
                                        max_features=10, min_samples_leaf= 1, min_samples_split=3, verbose=2)

        # Train it
        clr_rf = clf_rf.fit(x_train,y_train)
        # Test it
        print(clf_rf.score(x_test, y_test))
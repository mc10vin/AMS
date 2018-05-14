# -*- coding: utf-8 -*-

# Title          : Automatic Model Selection
# Author         : 김은종
# Origin_Date    : 05/11/2018
# Revision_Date  : 05/14/2018
# Version        : '0.1.2'


import h2o
from h2o.automl import H2OAutoML

h2o.init()

# load_file Definition Example
df = h2o.import_file(path='C:/workspace/python/creditcard.csv', destination_frame="df")
print(df.head())

# Input parameters that are going to train
response_column = 'Amount'
print(response_column)

training_columns = df.columns.remove(response_column)
print(training_columns)
# Output parameter train against input parameters
# TODO: from h2o.estimators.deeplearning import H2OAutoEncoderEstimator (H2OAutoEncoder)

# Split data into train and testing
train, test = df.split_frame(ratios=[0.8])

response_type = input("변수 형태를 입력해주세요. (default/binary): ")
# TODO: try-catch

# For binary classification, response should be a factor
if response_type == 'binary':
    train[response_column] = train[response_column].asfactor()
    test[response_column] = test[response_column].asfactor()

### AutoML

# Time to run the experiment
run_automl_for_seconds = int(input("최대 허용 시간을 .(s): "))
# TODO: try-catch

# RUN AutoML
aml = H2OAutoML(max_runtime_secs=run_automl_for_seconds)

aml.train(x=training_columns, y=response_column,
          training_frame=train,
          leaderboard_frame=test)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb)

# The leader model is stored here
aml.leader
print(aml.leader)

# predict
pred = aml.predict(test_data=test)

pred_df = pred.as_data_frame()
pred_df.to_csv('C:/workspace/python/creditcard_result.csv', header=True, index=False)

""" for version 0.3.1


if data==numerical:
    Do nothing
elif data==categorical:
    Label encoding
    one-hot encoding
elif data==text:
    counts
    tf-idf

# Label encoding
# from sklearn.preprocessing import LabelEncoder
#
# lbl_enc = LabelEncoder()
# lbl_enc.fit(xtrain[categorical_features])
# xtrain_cat = lbl_enc.transform(xtrain[categorical_features])
#
# One-hot encoding
# from sklearn.preprocessing import OneHotEncoder
#
# ohe = OneHotEncoder()
# ohe.fit(xtrain[categorical_features])
# xtrain_cat = ohe.transform(xtrain[categorical_features])


from sklearn.feature_extraction.text import TfidfVectorizer

Tfv = TfidfVectorizer(min_df=3, max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,',
                      ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1,
                      stop_words='english')

import numpy as np
from scipy import sparse

# In case of dense data
X = np.hstack([x1, x2, x3])

# In case data is sparse
X = sparse.hstack([x1, x2, x3])

from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

pca = PCA(n_compoents=10)
skb = SelectKBest(k=1)
combined_features = FeatureUnion([('pca', pca), ('skb', kskb)])

pca = PCA(n_components=12)
pca.fit(xtrain)
xtrain = pca.transform(xtrain)

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=128)
svd.fit(xtrain)
xtrain = svd.transform(xtrain)
# Random Forest Feature Selection
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X, y)
X_selected = clf.transform(X)

# Feature importances from GBM (XGBOOST)
import xgboost as xgb

params = {}

model = xgb.train(params, xtrain, num_boost_round=100)
sorted(model.get_fscore().items(), key=lambda t: -t[1])

# Chi2 feature election
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

skb = SelectKBest(chi2, k=20)
skb.fit_transform(X, y)


# Greedy feature selection

def selectionLoop(self, X, y):
    score_history = []
    good_features = set([])
    num_features = X.shape[1]

    while len(score_history) < 2 or score_history[-1][0] > score_history[-2][0]:
        scores = []
        for feature in range(num_features):
            if feature not in good_features:
                selected_features = list(good_features) + [feature]

                Xts = np.column_stack(X[:, j] for j in selected_features)

                score = self.evaluateScore(Xts, y)
                scores.append((score, feature))

                if self._verbose:
                    print('Current AUC :', np.mean(score))

            good_features.add(sorted(scores)[-1][1])
            score_history.append(sorted(scores)[-1])
            if self._verbose:
                print('Current Features : ', sorted(list(good_features)))

        # Remove last added feature
        good_features.remove(score_history[-1][1])
        good_features = sorted(list(good_features))
        if self._verbose:
            print('Selected Features : ', good_features)
"""

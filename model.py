#Import all the libraries

import pandas as pd
import numpy as np
import gensim
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
import pickle

nltk.download('punkt_tab')

train_df = pd.read_csv(r'train.csv')
data = train_df['text'].to_list()

# preproces the documents, and create TaggedDocuments
tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),tags=[str(i)]) for i,doc in enumerate(data)]

# train the Doc2vec model
model = Doc2Vec(vector_size=50,min_count=2, epochs=50)
model.build_vocab(tagged_data)
model.train(tagged_data,total_examples=model.corpus_count,epochs=model.epochs)

#Saving the tokenizer model
pickle.dump(model,open('model.pkl','wb'))



# get the document vectors
document_vectors = [model.infer_vector(word_tokenize(doc.lower())) for doc in data]

y = train_df['target'].to_list()

#Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(document_vectors, y, random_state = 42, stratify = y, test_size = 0.2)

#Initialize an object of the XGBClassifier
xgb = XGBClassifier(random_state = 42, learning_rate = 0.1)

#hyperparameters to tune
param_grid = {
    'n_estimators' : [300,500],
    'max_depth' : [20,25],
    'min_child_weight' : [2,5,10]
}

#grid search to optimise hyperparameters
grid_search = GridSearchCV(estimator = xgb, param_grid = param_grid, cv = 5, n_jobs = -1, verbose =2)

# Fit grid search on training data
grid_search.fit(X_train, y_train)

#Use the best estimator for prediction
xgb_best = grid_search.best_estimator_

#Saving the classifier
pickle.dump(xgb_best, open('xgb_best.pkl','wb'))
"""
sample execution command:
> python train_classifier.py ../data/s0umDisasterResDB.db classifier.pkl

Arguments:
    1) SQLite db path (containing pre-processed data)
    2) pickle file name to save ML model
"""

import sys
import pandas as pd
import numpy as np
import pickle

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import fbeta_score, classification_report
from scipy.stats.mstats import gmean

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    Load the database
    
    :param database_filepath: (str) path to SQLite db

    :return X: feature DataFrame
    :return Y: label DataFrame
    :return category_names: used for data visualization (app)
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df', engine)
    X = df['message']
    Y = df.iloc[:, 4:]

    Y['related'] = Y['related'].map(lambda x: 1 if x == 2 else x)

    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize function
    
    :param text: list of text messages (english)

    :return clean_tokens: tokenized text, clean for ML modeling
    """

    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build Model function
    
    This function output is a SciKit ML Pipeline that process text messages
    according to NLP best-practice and apply a classifier.

    :return model: SciKit ML Pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
    ], verbose=1)

    # hyper-parameter grid
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }

    # create model
    model = GridSearchCV(estimator=pipeline,
                         param_grid=parameters,
                         verbose=3,
                         cv=3)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies ML pipeline to a test set and prints out
    model performance (accuracy and f1score)
    
    Arguments:
    :param model: SciKit ML Pipeline
    :param X_test: test features
    :param Y_test: test labels
    :param category_names: label names (multi-output)

    """

    y_pred = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
    :param model: GridSearchCV or SciKit Pipeline object
    :param model_filepath: (str) destination path to save .pkl file
    
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle

    :arg : the file path of the messages dataset
    :arg : the file path of the categories dataset
    :arg : the database path that the data will be saved
    """

    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

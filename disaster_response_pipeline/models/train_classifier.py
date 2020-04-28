import sys
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine


import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

# import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def load_data(database_filepath):
    """
    Loads data from the sqlite database with path passed in. 

    Args: 
        database_filepath: the path of the database file
    Returns: 
        X (DataFrame): Messages 
        Y (DataFrame): One-hot encoded categories
        categories (List)
    """

    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    categories = Y.columns

    return X, Y, categories


def tokenize(text):
    """
        Processes and tokenizes the message by: 
        - replacing urls
        - converting to lower cases (normalization)
        - remove stopwords
        - stripping white spaces
    Args: 
        text: input messages
    Returns: 
        cleaned tokens(List)
    """
    # Define url regex
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Find and replace urls
    found_urls = re.findall(url_regex, text)
    for url in found_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize sentences
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # save cleaned tokens
    clean_tokens = [lemmatizer.lemmatize(
        token).lower().strip() for token in tokens]

    # remove stopwords
    STOPWORDS = list(set(stopwords.words('english')))
    clean_tokens = [token for token in clean_tokens if token not in STOPWORDS]

    return clean_tokens


def build_model():
    """
      Builds pipeline - CountVectorizer, TfidfTransformer, MultiOutputClassifier, 
      Grid Search
    Args: 
        None
    Returns: 
        Classifier Object
    """
    # create pipleline
    pipeline = Pipeline([
        ('vec', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100)))
    ])

    # Grid Search
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__max_features': ['sqrt', 0.5]}

    cv = GridSearchCV(estimator=pipeline,
                      param_grid=parameters, cv=5, n_jobs=10)

    return cv


def evaluate_model(model, X_test, Y_test, categories):
    """
    Evaluate the model's results, using precison, recall and f1-score

    Args: 
        model: the model to be evaluated
        X_test: X_test dataframe
        Y_test: Y_test dataframe
        categories: category names list defined in load data
    Returns: 
        results (DataFrame)
    """
    # predict on the X_test dataframe
    y_pred = model.predict(X_test)

    # build classification report on every column
    results = []
    for i in range(len(categories)):
        results.append([recall_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                        precision_score(
                            Y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                        f1_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro')])
    # build result dataframe
    results = pd.DataFrame(results, columns=['precision', 'recall', 'f1 score'],
                           index=categories)
    return results


def save_model(model, model_filepath):
    """
        Save model to pickle
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, categories = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categories)

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

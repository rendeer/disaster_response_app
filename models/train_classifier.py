import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import numpy as np
import pickle

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
#from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import warnings

warnings.simplefilter('ignore')


def load_data(database_filepath):
    """
    Load and merge messages and categories datasets
    
    Params:
    database_filename: string. Filename for SQLite database containing cleaned message data.
       
    Returns:
    X: dataframe. Dataframe containing features dataset.
    Y: dataframe. Dataframe containing labels dataset.
    category_names: list of strings. List containing category names.
    """
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    # Create list containing all category names
    category_names = list(Y.columns.values)
    
    return X, Y, category_names


def tokenize(text):
    """
    apply normalization, tokenization and stemming
    
    input: text to process
    output: a list of normalized and stemmed word tokens
    """
    # convert to lowercases and remove punctuation
    text = text.lower()
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)
    
    #tokenize words
    tokens = word_tokenize(text)
    
    # stem words and remove stopwords
    stemmer = PorterStemmer()
    
    return [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]

def multiclass_f1_score(y_true, y_pred):
    """
    Calculate mean F1 score for all of the output classifiers
    
    Params:
    y_true: array. actual labels.
    y_pred: array. predicted labels.
        
    output:
    score: float. mean F1 score for all of the output classifiers
    """
    f1_list = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i], average='micro')
        f1_list.append(f1)
        
    score = np.mean(f1_list)
    return score


def build_model():
    """
    Build a machine learning pipeline
    
    Params:
    None
       
    Returns:
    cv: gridsearchcv object. Gridsearchcv object that transforms the data, creates the 
    model object and finds the optimal model parameters.
    """
    # Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize, min_df = 5)),
        ('tfidf', TfidfTransformer(use_idf = True)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10,
                                                             min_samples_split = 10)))
    ])
    
    # Create parameters dictionary
    parameters = {'vect__min_df': [1, 5],
                  'tfidf__use_idf':[True, False],
                  'clf__estimator__n_estimators':[10, 25], 
                  'clf__estimator__min_samples_split':[2, 5, 10]}
    
    # Create scorer
    scorer = make_scorer(multiclass_f1_score)
    
    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer, verbose = 10)
    return cv

def get_eval_metrics(ground_truth, predicted, col_names):
    """
    Calculate evaluation metrics for ML model
    
    params:
    ground_truth: array. actual labels
    predicted: array. predicted labels.
    col_names: list. list containing names for each of the predicted fields.
       
    output:
    metrics_df: dataframe. Dataframe containing the precision, recall 
    and f1 score for a given set of actual and predicted labels.
    """
    metrics = []
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        precision = precision_score(ground_truth[:, i], predicted[:, i], average='micro')
        recall = recall_score(ground_truth[:, i], predicted[:, i], average='micro')
        f1 = f1_score(ground_truth[:, i], predicted[:, i], average='micro')
        
        metrics.append([precision, recall, f1])
    
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Precision', 'Recall', 'F1'])
      
    return metrics_df


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Returns test precision, recall and F1 score for fitted model
    
    Params:
    model: model object. Fitted model object.
    X_test: dataframe. Dataframe containing test features dataset.
    Y_test: dataframe. Dataframe containing test labels dataset.
    category_names: list of strings. List containing category names.
    
    Returns:
    None
    """
    # Predict labels for test dataset
    Y_pred = model.predict(X_test)
    
    # Calculate and print evaluation metrics
    eval_metrics = get_eval_metrics(np.array(Y_test), Y_pred, category_names)
    print(eval_metrics)


def save_model(model, model_filepath):
    """
    fitted model
    
    Params:
    model: model object. Fitted model object.
    model_filepath: string. Filepath for where fitted model should be saved
    
    Returns:
    None
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
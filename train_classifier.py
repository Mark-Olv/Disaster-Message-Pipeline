import sys
import os
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
nltk.download('stopwords')
from nltk.corpus import stopwords


def load_data(database_filepath):
    """
    Loads the data from the database and separates the data into input and
    output of the model for the training and testing steps. Additionally, 
    ouputs the categories names so it can be used in the evaluation step.
    
    Keyword Argument:
        database_filepath - A filepath of where to find the database file
    
    Returns:
        X - A pandas series object.
        Y - A pandas dataframe.
        Y.columns - A python list.
    """
    complete_database_filepath = 'sqlite:///' + database_filepath
    engine = create_engine(complete_database_filepath)
    # When you save a db as tables_name.db, the tables_name is the table's name
    table_name = os.path.basename(database_filepath).replace('db', '').replace('.', '')
    sql = 'SELECT * FROM ' + table_name
    df = pd.read_sql(sql, engine.connect())
    df.drop('original', axis=1, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)

    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y, Y.columns

def tokenize(text):
    """
    Used in the model. Cleans a message of extraneous words and breaks down
        the message into usable components.
    
    Keyword Argument:
        text - A string
        
    Returns:
        clean_tokens - A list
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
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

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """A class for customizing the model. This is a custom step implemented 
    in our model. Sklearn's pipeline requires the fit and transform attributes
    to work with a pipeline object. This class was obtained from Udacity.
    
    Methods:
        starting_verb - uses nltk to process data into numerical data based on
        the part of speech the word belongs to.
        fit - passes itself through. 
        transform - Applies the starting_verb function to the message series
        
    """
    def starting_verb(self, text):
        """
        Uses nltk to process data into numerical data based on
        the part of speech the word belongs to.
        """
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)
        
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(tokenize(sentence))

            # index pos_tags to get the first word and part of speech tag
            first_word, first_tag = pos_tags[0]
            
            # return true if the first word is an appropriate verb or RT for retweet
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1

            return 0


    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)

        return pd.DataFrame(X_tagged)


def build_model():
    """
    Builds a pipeline 
    
    Keyword arguments:
        None
    Return:
        pipeline - A Scikit learn pipeline.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=5, 
                                                             n_jobs=-1, verbose=True)))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model.
    
    Keyword argument:
        model - A pipeline object.
        X_test - Panda's series.
        Y_test - Panda's dataframe.
        category_names - A python list.
    Return:
        None
        prints sklearn's metric classification report
    """
    Y_pred_test = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred_test, target_names=category_names))



def save_model(model, model_filepath):
    """
    Saves the model into a pickle file.
    
    Keyword arguemnts:
        model - A pipeline object
        model_filepath - A filepath to save the model at.
    Return:
        None
    """
    
    #dump and dumps are different pickle functions.
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        #X_train = X_train.astype('str')
        
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
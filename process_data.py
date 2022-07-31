import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df
    

def clean_data(df):
    
    categories_id = df['id'].astype('int64').copy()
    categories = df['categories'].str.split(';', expand=True)

    a = categories.iloc[0:1,:].transpose().copy()#str[0:-2]
    a.rename({0:'column_names'}, axis=1, inplace=True)
    a.columns
    a['column_names'] = a['column_names'].str[0:-2] # maybe this step
    category_colnames = list(a['column_names'])
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]  
    
    # maybe doesn't like this step
    
    # convert column from string to numeric
        categories[column] = categories[column].astype('int64')
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    categories = pd.concat([categories_id, categories], axis=1)

    df = df.merge(categories, how='inner', left_on='id', right_on='id')
    df.drop_duplicates(inplace=True)
    df.drop(df[df['message'].duplicated()].index, inplace=True)
    df = df[~df['message'].str.isspace()]
    df.reindex(copy=False)
    df[df['related'] == 2] = 1
    df[df['request'] == 2] = 1
    
    return df


def save_data(df, database_filename):
    db_name = 'sqlite:///' + database_filename
    engine = create_engine(db_name)
    df.to_sql(database_filename, engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print(df.info())

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
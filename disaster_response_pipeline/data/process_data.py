import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
        Loads data from csv files passed as input and merges them. 
    Args: 
        messages_filepath: path of the messages.csv files to be loaded
        categories_filepath: path of the categories.csv file to be loaded
    Returns: 
        df (DataFrame): messages and categories merged as one dataframe
    """
    # read csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge messages and categories dataframes
    df = pd.merge(messages, categories, on = 'id')
    return df

def clean_data(df):
     """
        Transforms loaded data
        1. Handles Duplicates by removing
        2. Renames columns          

    Args: 
        df: dataframe to be cleaned

    Returns: 
        df (DataFrame): cleaned dataframe
    """
    # split categories columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[1]

    # extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.split('-')[0])

    # rename columns of categories
    categories.columns = category_colnames

    # set each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]))

   # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)

    # join the original dataframe with the new `categories` dataframe
    df = df.join(categories)

    # drop duplicates
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
      """
        load processed dataframe into sqlite database

    Args: 
        df: The preprocessed dataframe
        database_filename: name of the database
    Returns: 
        None
    """

    # save data into a sqlite database
    engine = create_engine('sqlite:///Messages.db')
    df.to_sql('Messages', engine, index=False, if_exists='replace')




def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

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
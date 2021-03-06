import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
      Function:
      load data from messages and categories csv files and then merge them

      Args:
      messages_filepath (str): the file path of the messages csv file
      categories_filepath (str): the file path of the categories csv file

      Return:
      df (DataFrame): A dataframe of messages and categories
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    """
      Function:
      clean the Dataframe df

      Args:
      df (DataFrame): A dataframe of messages and categories to be cleaned

      Return:
      df (DataFrame): A clean dataframe of messages and categories
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.rstrip('- 0 1'))
    print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 and 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], errors='coerce')

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(subset=['message'], inplace=True)
    df = df[df.related != 2]

    return df

def save_data(df, database_filename):
    """
       Function:
       Save the Dataframe df in a database
       Args:
       df (DataFrame): A dataframe of messages and categories
       database_filename (str): The file name of the database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_messages_tbl', engine, index=False, if_exists = 'replace')  


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
"""
sample execution command:
> python process_data.py disaster_messages.csv disaster_categories.csv s0umDisasterResDB.db

Arguments:
    1) CSV file containing messages (disaster_messages.csv)
    2) CSV file containing categories (disaster_categories.csv)
    3) SQLite destination database (s0umDisasterResDB.db)
"""

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories dataset and merge the two datasets

    :param messages_filepath: (str) path to messages csv file
    :param categories_filepath: (str) path to categories csv file

    :return df: DataFrame that merges the two datasets
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    """
    Cleans the DataFrame

    :param df: raw data Pandas DataFrame

    :return df: clean data Pandas DataFrame
    """

    # create a DataFrame of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    # select the first row of the categories DataFrame
    row = categories.iloc[0,:].values
    # use this row to extract a list of new column names for categories.
    new_cols = [r[:-2] for r in row]
    # rename the columns of `categories` DataFrame
    categories.columns = new_cols

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from 'df' DataFrame
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original DataFrame with the new 'categories' DataFrame
    df[categories.columns] = categories
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save DataFrame to sql database
    
    :param df: Clean data Pandas DataFrame
    :param database_filename: (str) database file (.db) destination path
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('df', engine, index=False, if_exists='replace')


def main():
    """
    Main Data Processing function,
    This function implements the ETL pipeline:
        1) Data extraction from .csv
        2) Data cleaning and pre-processing
        3) Data loading to SQLite database

    :arg : the file path of the messages dataset
    :arg : the file path of the categories dataset
    :arg : the database path that the data will be saved
    """

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
        print('Please provide the file paths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

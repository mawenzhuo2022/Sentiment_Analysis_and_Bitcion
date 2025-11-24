# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/20
# @Function: Text Preprocessing module

from typing import List
import pandas as pd
import re

import emoji



def remove_url(text: str) -> str:
    """
    Remove URLs from text
    :param text: the text to be processed
    :return: text with URLs removed
    """
    if pd.isna(text):  # Handle NaN values
        return ""
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "", str(text))


def remove_emoji(text: str) -> str:
    """
    Remove emoji from text
    :param text: the text to be processed
    :return: text with emoji removed
    """
    if pd.isna(text):
        return ""
    return emoji.replace_emoji(str(text), "")


def remove_special_cha(text: str) -> str:
    """
    Remove special characters while keeping:
    - Alphanumeric characters
    - Spaces
    - Basic punctuation (.,!?-;:'"())
    - Currency and numerical symbols ($%#@)
    - Plus and minus signs (+-)

    :param text: the text to be processed
    :return: text with special characters removed
    """
    if pd.isna(text):
        return ""
    # Keep these characters:
    # \w - word characters (letters, numbers, underscore)
    # \s - whitespace
    # .,!?-;:'"() - basic punctuation
    # $%#@ - common symbols
    # +- - arithmetic operators
    special_char_pattern = r"[^\w\s.,!?;:'\"\(\)$%#@+-]"
    return re.sub(special_char_pattern, "", str(text))


def clean_text(text: str) -> str:
    """
    Remove URLs, emoji and special characters from text
    :param text: the text to be processed
    :return: cleaned text
    """
    cleaned_text = remove_url(text)
    cleaned_text = remove_emoji(cleaned_text)
    cleaned_text = remove_special_cha(cleaned_text)
    return cleaned_text



def text_preprocessing(df):

    # Get string columns
    string_columns = df.select_dtypes(include='object').columns
    print(string_columns)
    # Apply cleaning to each string column
    for column in string_columns:
        df[column] = df[column].apply(clean_text)
    df= df.dropna()
    return df

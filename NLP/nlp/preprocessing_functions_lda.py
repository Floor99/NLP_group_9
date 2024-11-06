from collections import Counter
import pandas as pd
import re
# import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# CONVERT TO LOWERCASE
def convert_to_lowercase(series:pd.Series) -> pd.Series:
    series = series.copy()
    return series.str.lower()

# REMOVE URLs STARTING WITH HTTP(S)
def remove_https_urls_from_string(text) -> str:
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.sub('', text)

# REMOVE DOMAIN-BASED URLs
def remove_domain_urls_from_string(series: pd.Series):
    domain_suffixes = ['com', 'net', 'org', 'edu', 'uk', 'de', 'nl', 'us']
    pattern = r'\w+\s+(?:' + '|'.join(domain_suffixes) + r')\b'
    return re.sub(pattern, '', series).strip() 

# REMOVE BOTH URL TYPES
def remove_all_urls(series: pd.Series) -> pd.Series:
    series = series.copy()
    series = series.apply(remove_https_urls_from_string)
    series = series.apply(remove_domain_urls_from_string)
    return series

# REMOVE ALL WORDS THAT ARE NOT CHARACTERS
def remove_non_words_characters(series:pd.Series) -> pd.Series:
    series = series.copy()
    return series.replace(to_replace=r'[^\w\s]', value='', regex=True)

# REMOVE DIGITS
def remove_digits(series:pd.Series) -> pd.Series:
    series = series.copy()
    return series.replace(to_replace=r'\d', value='', regex=True)

# REMOVE DIGITS AND WORDS WITH DIGITS
def remove_digits_and_words_with_digits(text):
    return re.sub(r'\w*\d\w*', '', text).strip()

def remove_any_digits_and_words_with_digits(series):
    return series.apply(remove_digits_and_words_with_digits)

# TOKENIZE WORDS
def tokenize_words(series:pd.Series) -> pd.Series:
    series = series.copy()
    return series.apply(lambda x: word_tokenize(x))
 
# REMOVE STOPSWORDS   
def remove_stopwords(series:pd.Series, language:str= "english") -> pd.Series:
    series = series.copy()
    return series.apply(lambda x: [word for word in x if word not in stopwords.words(language)])

# LEMMATIZE
def lemmatize(series:pd.Series) -> pd.Series:
    series = series.copy()
    return series.apply(lambda x:lemmatize_txt(x))

def lemmatize_txt(text:str) -> str:
    wordnet_lemmatizer = WordNetLemmatizer()
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

# REMOVE MOST AND LEAST USED WORDS
def remove_top_n_percentage_words(texts, n=0):
    processed_texts = texts.copy()
    word_freq = Counter([word for sentence in processed_texts for word in sentence])

    if n > 0:
        top_n = set(word for word, _ in word_freq.most_common(int(n/100 * len(word_freq))))
        processed_texts = [[word for word in sentence if word not in top_n] for sentence in processed_texts]
    return processed_texts

def remove_bottom_n_percentage_words(texts, n=0):
    processed_texts = texts.copy()
    word_freq = Counter([word for sentence in processed_texts for word in sentence])

    if n > 0:
        bottom_n = set(word for word, _ in word_freq.most_common()[:-int(n/100 * len(word_freq)) - 1:-1])
        processed_texts = [[word for word in sentence if word not in bottom_n] for sentence in processed_texts]
    return processed_texts

# REMOVE SINGLE LETTER WORD
def remove_single_letter_words(series:pd.Series) -> pd.Series:
    series = series.copy()
    return series.apply(lambda x: [word for word in x if len(word) > 1])


    

from collections import Counter
import pandas as pd
import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')

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
  
# TOKENIZE WORDS
def tokenize_words(series:pd.Series) -> pd.Series:
    series = series.copy()
    return series.apply(lambda x: word_tokenize(x))
 
# REMOVE STOPSWORDS   
def remove_stopwords(series:pd.Series, language:str= "english") -> pd.Series:
    series = series.copy()
    return series.apply(lambda x: [word for word in x if word not in stopwords.words(language)])

# LEMMATIZE WORDS 
# def lemmatize_text(text: str) -> str:
#     text = ' '.join(text)
#     doc = nlp(text) 
#     lemmatized_words = [token.lemma_ for token in doc]  
#     return lemmatized_words
    
# def lemmatize_words(series:pd.Series) -> pd.Series:
#     series = series.copy()
#     return series.apply(lemmatize_text)

def lemmatize(series:pd.Series) -> pd.Series:
    series = series.copy()
    return series.apply(lambda x:lemmatize_txt(x))

def lemmatize_txt(text:str) -> str:
    wordnet_lemmatizer = WordNetLemmatizer()
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text


### Word frequency 
def get_word_frequency(series:pd.Series):
    all_words_after_preprocessing = [l for ls in series for l in ls]
    counter = Counter(all_words_after_preprocessing)
    word_count = counter.most_common()
    return word_count

def n_most_frequent_words(series:pd.Series, n_words):
    all_words = [word for sublist in series for word in sublist]

    fdist = FreqDist(all_words)
    
    df_fdist = pd.DataFrame({'Word': fdist.keys(),
                             'Frequency': fdist.values()})
    df_fdist = df_fdist.sort_values(by='Frequency', ascending=False)
    
    most_freq_words_list = list(df_fdist['Word'][0:n_words])
    
    return most_freq_words_list

def multiple_word_remove_func(series:pd.Series, words_2_remove_list):    
    cleaned_series = series.apply(
        lambda word_list: [word for word in word_list if word not in words_2_remove_list]
    )
    
    return cleaned_series

def n_times_least_used(series:pd.Series, n_times):
    all_words = [word for sublist in series for word in sublist]
    fdist = FreqDist(all_words)
    df_fdist = pd.DataFrame({'Word': fdist.keys(),
                             'Frequency': fdist.values()})
    df_fdist = df_fdist.sort_values(by='Frequency', ascending=False)
    
    least_used = df_fdist.loc[df_fdist['Frequency'] <= n_times]

    return least_used

def get_number_of_words_with_up_to_n_times_present(word_frequency, n):
    least_common = sum(1 for t in word_frequency if 1 <= t[1] <= n)
    return least_common









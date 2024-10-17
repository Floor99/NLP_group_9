import pandas as pd
import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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
def remove_domain_urls_from_string(article):
    domain_suffixes = ['com', 'net', 'org', 'edu', 'uk', 'de', 'nl', 'us']
    pattern = r'\w+\s+(?:' + '|'.join(domain_suffixes) + r')\b'
    return re.sub(pattern, '', article).strip() 

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
def lemmatize_text(text: str) -> str:
        doc = nlp(text)  
        lemmatized_words = [token.lemma_ for token in doc]  
        return ' '.join(lemmatized_words)
    
def lemmatize_words(series:pd.Series) -> pd.Series:
    return series.apply(lemmatize_text)
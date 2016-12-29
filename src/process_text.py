from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
import string
import re
from urllib.parse import *

from load_data import expanded_urls


tweet_url_regex = r'http\S+'
stop_words = stopwords.words('english')
tokenizer = TweetTokenizer()
stemmer = PorterStemmer()


def process(tweet):
    t = tokenize(tweet)
    t = filter_tokens(t)
    t = process_tokens(t)
    return t


def tokenize(tweet):
    # return tweet.split()
    return tokenizer.tokenize(tweet)


def filter_tokens(tokens, minlen=3, remove_stopwords=True, remove_punct=True):
    filtered = [t for t in tokens if len(t) >= minlen]
    if remove_stopwords:
        filtered = [t for t in filtered if t not in stop_words]
    if remove_punct:
        filtered = [t for t in filtered if t not in string.punctuation]

    return filtered


def process_tokens(tokens):
    return [process_token(t) for t in tokens]


def process_token(token, stem=True, remove_mentions=True):
    t = token.lower()
    if not re.match(tweet_url_regex, token):
        t = stemmer.stem(t)
        t = re.sub('\d+', '', t)
        if remove_mentions and t.startswith('@'):
            t = ''
    return t


def replace_map_url(text: str) -> str:
    urls = re.findall(tweet_url_regex, text)
    for url in urls:
        if url in expanded_urls:
            cleanurl = clean_url(expanded_urls[url])
            text = text.replace(url, cleanurl)
    return text


def clean_url(url):
    spl = urlsplit(url)
    spl = urlsplit(spl.geturl())
    return urlunsplit((spl[0], spl[1], spl[2], '', ''))


if __name__ == '__main__':
    tweet = 'Oscar #Pistorius weeps, vomits as grisly Reeva Steenkamp autopsy details come out in court, by @MarieLouiseCNN and me http://t.co/GIOBRyYVNJ'

    print(tweet)
    print(process(tweet))

    print(replace_map_url(tweet))

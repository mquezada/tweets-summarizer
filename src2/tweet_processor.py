from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer

from collections import defaultdict
from urllib.parse import urlsplit, urlunsplit
import re
import string


class TweetProcessor:
    def __init__(self, data_loader):
        self.data = data_loader
        self.tokenizer = TweetTokenizer()
        self.stemmer = PorterStemmer()
        self.stopwords = stopwords.words('english')
        self.re_url = r'http\S+'
        self.punctuation = string.punctuation
        self.vocab = defaultdict(set)

    def __iter__(self):
        yield from self.process_tweet()

    def process_tweet(self):
        for tokens in self.token_generator():
            processed_tweet = []
            for token in tokens:
                processed_token, tag = self.process_token(token)
                if processed_token:
                    processed_tweet.append((processed_token, tag))
            if processed_tweet:
                yield processed_tweet

    def token_generator(self):
        for text in self.data.corpus['text']:
            yield self.tokenizer.tokenize(text)

    def process_token(self, token):
        original = token

        if re.match(self.re_url, token):
            url = self.data.expanded_urls.get(token, token)
            return TweetProcessor.clean_url(url), 'URL'

        token = token.lower()
        if token in self.stopwords or token in self.punctuation:
            return None, None

        if token.startswith('@'):
            return None, None

        token = token.translate({ord(k): "" for k in self.punctuation})
        #token = self.stemmer.stem(token)
        self.vocab[token].add(original)

        return token, self.data.token_tags.get(original, "NA")

    @staticmethod
    def clean_url(url):
        spl = urlsplit(url)
        spl = urlsplit(spl.geturl())
        return urlunsplit((spl[0], spl[1], spl[2], '', ''))

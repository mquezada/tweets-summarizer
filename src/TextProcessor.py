from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from nltk.tag.stanford import StanfordPOSTagger

#import gensim

from collections import defaultdict, Counter
from urllib.parse import urlsplit, urlunsplit
import re
import string


class TextProcessor:
    def __init__(self, corpus, expanded_urls):
        self.tokenizer = TweetTokenizer()
        self.stemmer = PorterStemmer()
        self.stopwords = stopwords.words('english')
        self.corpus = corpus
        self.expanded_urls = expanded_urls
        self.re_url = r'http\S+'
        self.punctuation = string.punctuation
        self.stanford_pos_pwd = '/Users/mquezada/stanford-postagger-full-2015-12-09/'
        self.stanford_pos = StanfordPOSTagger(self.stanford_pos_pwd + 'models/english-left3words-distsim.tagger',
                                              self.stanford_pos_pwd + 'stanford-postagger.jar')
        self.tag_vocab = defaultdict(Counter)
        self.tag_token = dict()
        self.vocab = defaultdict(set)
        self.tags = Counter()

    def __iter__(self):
        yield from self.process()

    def process(self):
        for tokens in self.stanford_pos.tag_sents(self.tokenseq_generator()):
        #for tokens in self.tokenseq_generator():
            res = []
            for token, tag in tokens:
            #for token in tokens:
                processed = self.process_token(token)
                if processed:
                    #most_similar = self.w2v.most_similar(token)
                    self.tag_vocab[processed].update({tag: 1})
                    self.tag_token[token] = tag
                    self.tags.update({tag: 1})

                    res.append(processed)
            if res:
                yield res

    @staticmethod
    def clean_url(url):
        spl = urlsplit(url)
        spl = urlsplit(spl.geturl())
        return urlunsplit((spl[0], spl[1], spl[2], '', ''))

    def process_token(self, token):
        if re.match(self.re_url, token):
            return TextProcessor.clean_url(self.expanded_urls.get(token, token))

        t = token.lower()
        #t = token

        if t in self.stopwords or t in self.punctuation:
            return None

        if len(t) < 3 or t.startswith('@'):
            return None

        if not t.startswith('#'):
            t = t.translate({ord(k): "" for k in self.punctuation})

        t = self.stemmer.stem(t)

        self.vocab[t].add(token)
        return t

    def tokenseq_generator(self):
        for text in self.corpus:
            yield self.tokenizer.tokenize(text)


if __name__ == '__main__':
    processor = TextProcessor(['a b c', 'd e f', 'h j x',
                               '#GusttosoTeama Pistorius Fired Gun in Restaurant, Convinced Pal to Take Blame: Testim... http://t.co/aX4ouIlbjC #GusttosoTeama'],
                              dict())

    for text in processor:
        print(text)
    #print(processor.tag_vocab)
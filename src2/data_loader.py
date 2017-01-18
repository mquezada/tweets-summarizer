import pandas as pd
from pathlib import Path
import logging


class DataLoader:
    def __init__(self, data_path, dataset_name, urls_name, stanford_path="", load_pos=True):
        self.path = data_path
        self.dataset_name = dataset_name
        self.urls_name = urls_name
        self.corpus = self.load_tweets()
        self.expanded_urls = self.load_urls()
        self.token_tags = set()

        self.stanford_path = stanford_path
        if load_pos:
            self.compute_pos()

    def load_tweets(self):
        logging.info("Loading tweets from '%s'..." % self.dataset_name)
        path = Path(self.path, self.dataset_name)
        df = pd.read_table(path.as_posix(), engine='python', sep=r'\t')
        logging.info("Loaded %d tweets" % df.shape[0])
        return df[df['text'].notnull()]

    def load_urls(self):
        logging.info("Loading URLs from '%s'..." % self.urls_name)
        path = Path(self.path, self.urls_name)
        with path.open() as f:
            urls = [line.split() for line in f.readlines()][1:]
        logging.info("Loaded %d pairs (URL, Expanded URL)" % len(urls))
        return dict(urls)

    def _get_pos_path(self):
        pos_name = self.dataset_name.split('.')[0] + "_pos.txt"
        return Path(self.path, pos_name)

    def compute_pos(self):
        path = self._get_pos_path()

        if not path.exists():
            logging.info("Computing POS tags from tweets...")
            from nltk.tag.stanford import StanfordPOSTagger
            from nltk.tokenize.casual import TweetTokenizer

            s_path = self.stanford_path

            stanford_tagger = StanfordPOSTagger(Path(s_path, 'models/english-left3words-distsim.tagger').as_posix(),
                                                Path(s_path, 'stanford-postagger.jar').as_posix())
            tokenizer = TweetTokenizer()

            tagged_tweets = stanford_tagger.tag_sents([tokenizer.tokenize(text)
                                                       for text in self.corpus['text']])

            for tagged_tweet in tagged_tweets:
                for token, tag in tagged_tweet:
                    if len(token) > 0:
                        self.token_tags.add((token, tag))

            with path.open('w') as f:
                for token, tag in self.token_tags:
                    f.write('%s\t%s\n' % (token, tag))

            self.token_tags = dict(self.token_tags)

            logging.info("Wrote %d unique pairs (word, pos_tag)" % len(self.token_tags))
        else:
            logging.info("POS tag file already exist. Loading into class instance...")
            with path.open() as f:
                tags = [line.split() for line in f.readlines()]
            self.token_tags = dict(tags)
            logging.info("POS tags loaded.")
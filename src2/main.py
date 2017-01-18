from data_loader import DataLoader
from tweet_processor import TweetProcessor
from settings import DATA_PATH, STANFORD_POS_PATH, W2V_PATH
import logging
import gensim
from operator import itemgetter

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


data = DataLoader(data_path=DATA_PATH,
                  dataset_name='oscar_pistorius.txt',
                  urls_name='urls_pistorius.txt',
                  stanford_path=STANFORD_POS_PATH)

tweets = TweetProcessor(data)
proc_tweets = [t for t in tweets]

logging.info("Loading w2v model...")
model = gensim.models.Word2Vec.load(W2V_PATH)
threshold = 0.8

for i in range(len(proc_tweets)):
    if i % 1000 == 0:
        logging.info("processed %d tweets" % i)
    tweet = proc_tweets[i]
    for word, tag in tweet:
        if word in model and tag.startswith('NN'):
            ms = model.most_similar(word, topn=3)
            selected = list(map(itemgetter(0), filter(lambda _, s: s > threshold, ms)))
            proc_tweets[i].extend(selected)
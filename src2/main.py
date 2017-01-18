from data_loader import DataLoader
from tweet_processor import TweetProcessor
from settings import DATA_PATH, STANFORD_POS_PATH, W2V_PATH
import logging
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


data = DataLoader(data_path=DATA_PATH,
                  dataset_name='oscar_pistorius.txt',
                  urls_name='urls_pistorius.txt',
                  stanford_path=STANFORD_POS_PATH)

tweets = TweetProcessor(data)
proc_tweets = [t for t in tweets]

logging.info("Loading w2v model...")
model = gensim.models.Word2Vec.load(W2V_PATH)


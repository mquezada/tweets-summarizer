import gensim
import numpy as np
from load_data import expanded_urls, df
from process_text import process, replace_map_url

from sklearn.cluster import KMeans


tweet_url_regex = r'http\S+'
ids_with_url = df.text.str.contains(tweet_url_regex, regex=True, na=False)
df_urls = df.loc[ids_with_url]


class ProcessedTweets:
    def __iter__(self):
        for text in df_urls['text']:
            yield process(replace_map_url(text))


sentences = ProcessedTweets()
model = gensim.models.Word2Vec(sentences)


urls = sorted(filter(lambda x: x.startswith('http'), model.vocab.keys()))

km = KMeans(n_clusters=5)
km.fit(model[urls])


for i in range(5):
    for j in range(len(model[urls][km.labels_ == i])):
        print(model.similar_by_vector(model[urls][km.labels_ == i][j], topn=1))
    print("=" * 10)
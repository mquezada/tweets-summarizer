import gensim
import numpy as np
from collections import namedtuple
from load_data import expanded_urls, df
from process_text import process, replace_map_url
from model_documents import docs
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from sklearn.cluster import KMeans

docs
tweet_url_regex = r'http\S+'
ids_with_url = df.text.str.contains(tweet_url_regex, regex=True, na=False)
df_urls = df.loc[ids_with_url]


class ProcessedTweets:
     def __iter__(self):
         for key, value in docs.items():
             yield process(replace_map_url(value))


sentences = ProcessedTweets()
docs_list = []
for i,value in enumerate(sentences):
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    words=value
    tags=[i]
    docs_list.append(analyzedDocument(words, tags))

model = gensim.models.Doc2Vec(docs_list)

print(len(docs))

print(model.most_similar('nepal',topn=5))
print(model.most_similar('food',topn=5))
print(model.doesnt_match('nepal food asdf'))


# urls = sorted(filter(lambda x: x.startswith('http'), model.vocab.keys()))
#
# km = KMeans(n_clusters=7)
# km.fit(model[urls])
#
# data = np.array([model[w] for w in model.vocab.keys()])
# words = [w for w in model.vocab.keys()]

# for i in range(5):
#     for j in range(len(model[urls][km.labels_ == i])):
#         print(model.similar_by_vector(model[urls][km.labels_ == i][j], topn=1))
#     print("=" * 10)


# print(model.most_similar('murder',topn=5))
# print(model.most_similar('pistoriu',topn=5))
# print(model.most_similar('trial',topn=5))
# print(model.most_similar('girlfriend',topn=5))
# print(model.most_similar('women',topn=5))

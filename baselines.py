import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


sw = stopwords.words('english')


def random_summarizer(data, n_tweets=10):
    return np.random.choice(data['text'], n_tweets, replace=False)


def process_text(text):
    tokens = text.split()
    tokens = [t.lower() for t in tokens if t not in sw]
    return tokens


def tfidf_summarizer(data, n_tweets=10):
    texts = data['text'].unique()
    vectorizer = TfidfVectorizer(tokenizer=process_text)
    t_matrix = vectorizer.fit_transform(texts)
    sim_matrix = t_matrix.dot(t_matrix.transpose())
    centrality = sim_matrix.sum(axis=0)  # sum by rows
    top_n_idx = centrality.argsort()[0, -n_tweets:].tolist()[0]

    return texts[top_n_idx]


if __name__ == '__main__':
    import pandas as pd
    data = '../../data/oscar_pistorius.txt'
    df = pd.read_csv(data, sep=r"\t", engine="python")
    df = df[df['is_retweet'] == 0]

    print(random_summarizer(df))

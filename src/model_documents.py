from load_data import df, expanded_urls

from collections import defaultdict
import numpy as np
import re


tweet_url_regex = r'http\S+'
ids_with_url = df.text.str.contains(tweet_url_regex, regex=True, na=False)
all_expanded = []


def get_url(row):
    return re.findall(tweet_url_regex, row['text'])


def replicate(row):
    url_list = get_url(row)
    return len([expanded_urls[u] for u in url_list if u in expanded_urls])


def map_url(row):
    url_list = get_url(row)
    all_expanded.extend([expanded_urls[u] for u in url_list if u in expanded_urls])


df_urls = df.loc[ids_with_url]
df_urls.apply(map_url, axis=1)

df_repl = df_urls.loc[np.repeat(df_urls.index.values, df_urls.apply(replicate, axis=1))]
df_repl['url_id'] = all_expanded


docs = dict()
for name, group in df_repl.groupby('url_id'):
    docs[name] = group['text'].str.cat(sep=' ')

if __name__ == '__main__':
    pass
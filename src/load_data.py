import pandas as pd

pd.options.display.max_colwidth = 140

DATA_DIR = '../../data/'
dataset_fn = 'nepal_earthquake.txt'
urls_fn = 'urls_nepal.txt'


df = pd.read_table(DATA_DIR + dataset_fn, engine='python', sep=r"\t")

with open(DATA_DIR + urls_fn) as f:
    urls = [line.split() for line in f.readlines()[1:]]

expanded_urls = {k : v for k, v in urls}
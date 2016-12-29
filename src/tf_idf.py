from model_documents import docs
from process_text import process

from sklearn.feature_extraction.text import TfidfVectorizer

urls = docs.keys()
contents = docs.values()

vectorizer = TfidfVectorizer(tokenizer=process)
tfidf_matrix = vectorizer.fit_transform(contents)
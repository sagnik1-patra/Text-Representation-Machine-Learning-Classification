from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import FeatureUnion
from feature_engineering import CustomFeatures

def make_vectorizer(kind='tfidf',max_features=10000,ngram_range=(1,2),**kwargs):
    cls=TfidfVectorizer if kind=='tfidf' else CountVectorizer
    if kind=='char': return CountVectorizer(analyzer='char',ngram_range=(2,4),max_features=max_features,**kwargs)
    return cls(max_features=max_features,ngram_range=ngram_range,**kwargs)

def with_custom_features(vectorizer):
    return FeatureUnion([('text',vectorizer),('custom',CustomFeatures())])

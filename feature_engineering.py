import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
POS=set('amazing excellent wonderful great best brilliant enjoyable beautiful fantastic perfect love loved superb'.split())
NEG=set('terrible awful worst boring poor bad horrible disappointing waste dull hated hate'.split())
class CustomFeatures(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        self.scaler_=StandardScaler()
        self.scaler_.fit(self._extract(X))
        return self
    def transform(self,X):
        return csr_matrix(self.scaler_.transform(self._extract(X)).astype('float32'))
    def _extract(self,X):
        rows=[]
        for s in X:
            s=str(s); words=re.findall(r"[a-zA-Z']+",s.lower()); n=max(len(words),1)
            p=sum(w in POS for w in words); q=sum(w in NEG for w in words)
            rows.append([len(s),len(words),sum(map(len,words))/n,len(re.findall(r'[.!?]+',s)),s.count('!'),sum(c.isupper() for c in s)/max(len(s),1),p,q,(p-q)/(p+q+1),int('amazing' in words),int('terrible' in words),int('worst' in words),int(bool(re.search(r'\b\d+(?:/10| stars?)\b',s.lower())))])
        return np.asarray(rows,dtype=np.float32)
FEATURE_NAMES=['characters','words','average_word_length','sentences','exclamations','uppercase_ratio','positive_words','negative_words','polarity_ratio','amazing','terrible','worst','rating_mention']

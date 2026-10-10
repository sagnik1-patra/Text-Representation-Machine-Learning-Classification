from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

def classifiers(seed=42):
    return {
      'MultinomialNB':MultinomialNB(),
      'BernoulliNB':BernoulliNB(),
      'LogisticRegression':LogisticRegression(max_iter=2000,random_state=seed),
      'LinearSVC':LinearSVC(max_iter=5000,random_state=seed),
      'RandomForest':RandomForestClassifier(n_estimators=120,max_depth=35,n_jobs=-1,random_state=seed),
      'GradientBoosting':HistGradientBoostingClassifier(max_iter=100,max_depth=5,random_state=seed),
      'KNN':KNeighborsClassifier(n_neighbors=7,metric='cosine',algorithm='brute',n_jobs=-1)
    }

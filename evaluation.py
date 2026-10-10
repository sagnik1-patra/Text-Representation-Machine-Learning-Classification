import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,confusion_matrix,classification_report

def scores(model,X,y):
    pred=model.predict(X)
    raw=model.decision_function(X) if hasattr(model,'decision_function') else model.predict_proba(X)[:,1] if hasattr(model,'predict_proba') else pred
    pm,rm,fm,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
    pw,rw,fw,_=precision_recall_fscore_support(y,pred,average='weighted',zero_division=0)
    return dict(accuracy=accuracy_score(y,pred),precision_macro=pm,recall_macro=rm,f1_macro=fm,precision_weighted=pw,recall_weighted=rw,f1_weighted=fw,roc_auc=roc_auc_score(y,raw)),pred,raw

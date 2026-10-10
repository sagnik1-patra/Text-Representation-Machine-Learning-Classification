import os,re,time,json,pickle,zipfile,argparse,warnings,html
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt,seaborn as sns,joblib,yaml
from scipy.sparse import issparse
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split,StratifiedKFold,RandomizedSearchCV,learning_curve,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_selection import SelectKBest,chi2,mutual_info_classif,SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,roc_curve,precision_recall_curve,classification_report
from sklearn.base import clone
from vectorization import make_vectorizer,with_custom_features
from models import classifiers
from evaluation import scores
from feature_engineering import FEATURE_NAMES
warnings.filterwarnings('ignore',category=FutureWarning)
ROOT=Path(r'C:\Users\sagni\Downloads\Text Representation & Machine Learning Classification')
SEED=42

def clean(s):
    s=html.unescape(str(s));s=re.sub(r'<[^>]*>',' ',s);s=re.sub(r'[^a-zA-Z\s\'!?]',' ',s)
    return re.sub(r'\s+',' ',s).strip()

def preprocess(s,lemmatizer=None):
    words=re.findall(r"[a-z']+",clean(s).lower())
    stops=ENGLISH_STOP_WORDS-set('no not nor never neither nothing nobody without against'.split())
    words=[w for w in words if w not in stops and len(w)>1]
    return ' '.join(lemmatizer.lemmatize(w) for w in words) if lemmatizer else ' '.join(words)

def plot_save(out,name):
    plt.tight_layout();plt.savefig(out/(name+'.png'),dpi=120,bbox_inches='tight');plt.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--quick',action='store_true');ap.add_argument('--no-lemma',action='store_true');args=ap.parse_args()
    root=args.root; root.mkdir(parents=True,exist_ok=True);out=root/'results';out.mkdir(exist_ok=True);art=root/'artifacts';art.mkdir(exist_ok=True)
    file=root/'IMDB Dataset.csv'
    if not file.exists(): raise FileNotFoundError(file)
    df=pd.read_csv(file,usecols=lambda c:c.lower() in ('review','sentiment'),dtype=str).dropna().drop_duplicates(subset=['review'])
    df.columns=[c.lower() for c in df.columns]
    if not {'review','sentiment'}<=set(df):raise ValueError('CSV must contain review and sentiment columns')
    df['label']=df.sentiment.str.lower().map({'positive':1,'negative':0});df=df.dropna(subset=['label']);df['label']=df.label.astype(int)
    if args.quick:df=df.groupby('label',group_keys=False).apply(lambda d:d.sample(min(1500,len(d)),random_state=SEED)).reset_index(drop=True)
    a,b=train_test_split(df,test_size=.4,stratify=df.label,random_state=SEED)
    v,t=train_test_split(b,test_size=.5,stratify=b.label,random_state=SEED)
    lemmatizer=None
    if not args.no_lemma:
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            nltk.download('wordnet',quiet=True);lemmatizer=WordNetLemmatizer();lemmatizer.lemmatize('movies')
        except Exception as e:print('WordNet unavailable, proceeding without lemmatization:',e)
    for part,d in [('train',a),('validation',v),('test',t)]:
        d=d.copy();d['processed']=d.review.map(lambda s:preprocess(s,lemmatizer));d.to_csv(out/f'{part}_preprocessed.csv.gz',index=False,compression='gzip')
        if part=='train':a=d
        elif part=='validation':v=d
        else:t=d
    x=a.processed.tolist();y=a.label.to_numpy();xv=v.processed.tolist();yv=v.label.to_numpy();xt=t.processed.tolist();yt=t.label.to_numpy()
    raw=a.review.tolist();rawv=v.review.tolist();rawt=t.review.tolist()
    pd.DataFrame({'train':a.label.value_counts(),'validation':v.label.value_counts(),'test':t.label.value_counts()}).fillna(0).plot(kind='bar');plt.title('Class distribution');plot_save(out,'01_class_distribution')
    # Vectorization parameter studies, training-only fits
    records=[];features=[];vocab=[];ngram=[]
    for n in [1000,5000,10000]:
        for md in [2,5,10]:
            for xd in [.8,.9,1.0]:
                for binary in [False,True]:
                    vec=make_vectorizer('bow',max_features=n,min_df=md,max_df=xd,binary=binary,ngram_range=(1,1))
                    z=vec.fit_transform(x);records.append({'max_features':n,'min_df':md,'max_df':xd,'binary':binary,'vocab_size':z.shape[1],'sparsity':1-z.nnz/(z.shape[0]*z.shape[1])})
    pd.DataFrame(records).to_csv(out/'bow_parameter_study.csv',index=False)
    pd.DataFrame(records).groupby('max_features').vocab_size.mean().plot(marker='o');plt.ylabel('Mean vocabulary size');plt.title('Vocabulary vs max_features');plot_save(out,'02_vocabulary_size')
    z=make_vectorizer('bow',max_features=1000).fit_transform(x[:min(100,len(x))]);sns.heatmap((z[:40,:80].toarray()>0).astype(int),cbar=False);plt.title('BoW nonzero feature heatmap');plot_save(out,'03_sparsity_heatmap')
    for norm in ['l1','l2',None]:
        for use_idf in [True,False]:
            for smooth in [True,False]:
                for sublinear in [True,False]:
                    vec=make_vectorizer('tfidf',max_features=5000,norm=norm,use_idf=use_idf,smooth_idf=smooth,sublinear_tf=sublinear)
                    z=vec.fit_transform(x);features.append({'norm':str(norm),'use_idf':use_idf,'smooth_idf':smooth,'sublinear_tf':sublinear,'vocab_size':z.shape[1],'mean_nonzero':float(z.data.mean()) if z.nnz else 0})
    pd.DataFrame(features).to_csv(out/'tfidf_parameter_study.csv',index=False)
    vec=make_vectorizer('tfidf',max_features=5000);z=vec.fit_transform(x);weights=np.asarray(z.mean(axis=0)).ravel();ix=np.argsort(weights)[-20:]
    plt.barh(vec.get_feature_names_out()[ix],weights[ix]);plt.title('Top 20 mean TF-IDF features');plot_save(out,'04_top_tfidf')
    plt.hist(z.data[:min(100000,len(z.data))],bins=45);plt.title('Nonzero TF-IDF score distribution');plot_save(out,'05_tfidf_distribution')
    zz=z[:3].toarray();ids=np.argsort(zz.sum(axis=0))[-15:];sns.heatmap(zz[:,ids],xticklabels=vec.get_feature_names_out()[ids],yticklabels=['Doc 1','Doc 2','Doc 3']);plt.xticks(rotation=60);plot_save(out,'06_sample_tfidf')
    ranges=[(1,1),(2,2),(3,3),(1,2),(1,3),(2,3)]
    for r in ranges:
        start=time.perf_counter();vec=make_vectorizer('bow',max_features=10000,ngram_range=r);z=vec.fit_transform(x)
        ngram.append({'range':str(r),'vocab_size':z.shape[1],'fit_seconds':time.perf_counter()-start,'sparsity':1-z.nnz/(z.shape[0]*z.shape[1])})
        if r in [(2,2),(3,3)]:
            ix=np.argsort(np.asarray(z.sum(axis=0)).ravel())[-20:][::-1];pd.DataFrame({'ngram':vec.get_feature_names_out()[ix],'count':np.asarray(z.sum(axis=0)).ravel()[ix]}).to_csv(out/f'top_{r[0]}grams.csv',index=False)
    vec=make_vectorizer('char',max_features=10000);start=time.perf_counter();z=vec.fit_transform(x)
    ngram.append({'range':'character (2,4)','vocab_size':z.shape[1],'fit_seconds':time.perf_counter()-start,'sparsity':1-z.nnz/(z.shape[0]*z.shape[1])})
    ng=pd.DataFrame(ngram);ng.to_csv(out/'ngram_study.csv',index=False);sns.barplot(data=ng,x='range',y='vocab_size');plt.xticks(rotation=45);plot_save(out,'07_ngram_vocabulary')
    # Train 7 classifier variants (6 families) and several vectorizers
    configs={'BoW_5k':make_vectorizer('bow',max_features=5000,ngram_range=(1,1)), 'TFIDF_5k':make_vectorizer('tfidf',max_features=5000,ngram_range=(1,1)), 'TFIDF_10k_12':make_vectorizer('tfidf',max_features=10000,ngram_range=(1,2)), 'Char_10k':make_vectorizer('char',max_features=10000)}
    rows=[];fitted={};best_score=-1;best_key=None
    for vn,vectorizer in configs.items():
        tic=time.perf_counter();z=vectorizer.fit_transform(x);zv=vectorizer.transform(xv);vector_seconds=time.perf_counter()-tic
        joblib.dump(vectorizer,art/f'{vn}_vectorizer.pkl',compress=3)
        for name,model in classifiers().items():
            try:
                # Gradient boosting needs dense input; use SVD to keep memory bounded.
                if name=='GradientBoosting':
                    from sklearn.decomposition import TruncatedSVD
                    reducer=TruncatedSVD(n_components=min(100,z.shape[1]-1),random_state=SEED)
                    zfit=reducer.fit_transform(z);zeval=reducer.transform(zv)
                else:reducer=None;zfit=z;zeval=zv
                tic=time.perf_counter();model.fit(zfit,y);train_s=time.perf_counter()-tic
                tic=time.perf_counter();metrics,pred,rawscores=scores(model,zeval,yv);infer_s=time.perf_counter()-tic
                key=vn+'__'+name
                row=dict(vectorization=vn,model=name,vectorization_seconds=vector_seconds,training_seconds=train_s,inference_seconds=infer_s,**metrics);rows.append(row)
                fitted[key]=(clone(vectorizer),model,reducer)
                if metrics['f1_macro']>best_score:best_score=metrics['f1_macro'];best_key=key
                if name in ['LogisticRegression','LinearSVC','MultinomialNB']:
                    joblib.dump({'model':model,'reducer':reducer},art/(key+'.pkl'),compress=3)
                print(key,round(metrics['accuracy'],4),flush=True)
            except Exception as e:print('SKIPPED',vn,name,str(e)[:180])
    result=pd.DataFrame(rows).sort_values('f1_macro',ascending=False);result.to_csv(out/'validation_comparison.csv',index=False)
    if result.empty:raise RuntimeError('All models failed')
    top=result.head(15);sns.barplot(data=top,x='accuracy',y='model',hue='vectorization');plt.xlim(0,1);plt.title('Validation accuracy');plot_save(out,'08_model_accuracy')
    sns.barplot(data=top,x='f1_macro',y='model',hue='vectorization');plt.xlim(0,1);plot_save(out,'09_model_f1')
    sns.barplot(data=top,x='training_seconds',y='model',hue='vectorization');plot_save(out,'10_training_times')
    # Five-fold randomized hyperparameter searches on three linear models, train only
    params={
      'MultinomialNB':{'classifier__alpha':[.01,.1,.5,1,2,5],'vectorizer__max_features':[5000,10000]},
      'LogisticRegression':{'classifier__C':[.1,.5,1,2,5,10],'classifier__penalty':['l1','l2'],'classifier__solver':['liblinear'],'vectorizer__max_features':[5000,10000]},
      'LinearSVC':{'classifier__C':[.05,.1,.5,1,2,5],'classifier__max_iter':[3000,6000],'vectorizer__max_features':[5000,10000]}}
    tuning=[];tuned={}
    for name,grid in params.items():
        pipe=Pipeline([('vectorizer',make_vectorizer('tfidf',max_features=10000,ngram_range=(1,2))),('classifier',classifiers()[name])])
        search=RandomizedSearchCV(pipe,grid,n_iter=6 if args.quick else 12,scoring='f1_macro',cv=StratifiedKFold(5,shuffle=True,random_state=SEED),n_jobs=2,random_state=SEED,refit=True)
        search.fit(x,y);m,_,_=scores(search.best_estimator_,xv,yv)
        tuning.append({'model':name,'cv_f1':search.best_score_,'validation_f1':m['f1_macro'],'best_params':str(search.best_params_)})
        tuned[name]=search.best_estimator_;joblib.dump(search.best_estimator_,art/(name+'_tuned.pkl'),compress=3)
        pd.DataFrame(search.cv_results_).to_csv(out/(name+'_cv_results.csv'),index=False)
    pd.DataFrame(tuning).to_csv(out/'hyperparameter_tuning.csv',index=False)
    # Feature engineering and selection ablation: fit only on training data
    ablation=[];ablation_models={}
    for tag,vectorizer in [('baseline',make_vectorizer('tfidf',max_features=10000,ngram_range=(1,2))),('custom',with_custom_features(make_vectorizer('tfidf',max_features=10000,ngram_range=(1,2))))]:
        pipe=Pipeline([('features',vectorizer),('classifier',LogisticRegression(max_iter=2000))]);pipe.fit(raw if tag=='custom' else x,y)
        m,_,_=scores(pipe,rawv if tag=='custom' else xv,yv);ablation.append({'experiment':tag,**m});ablation_models[tag]=pipe
    base=make_vectorizer('tfidf',max_features=10000,ngram_range=(1,2));z=base.fit_transform(x);zv=base.transform(xv)
    for tag,selector in [('chi2',SelectKBest(chi2,k=min(3000,z.shape[1]))),('l1',SelectFromModel(LogisticRegression(penalty='l1',solver='liblinear',C=.3,max_iter=2000))),('mutual_info',SelectKBest(lambda X,y:mutual_info_classif(X,y,discrete_features=False,random_state=SEED),k=min(1000,z.shape[1])))]:
        # MI on high-dimensional sparse features is expensive; sample a smaller training subset
        if tag=='mutual_info':
            sub=np.random.default_rng(SEED).choice(len(y),size=min(2000,len(y)),replace=False)
            selector.fit(z[sub].toarray()[:,:min(1500,z.shape[1])],y[sub]);z2=z[:,:min(1500,z.shape[1])];zv2=zv[:,:min(1500,z.shape[1])]
        else:z2=z;zv2=zv;selector.fit(z2,y)
        model=LogisticRegression(max_iter=2000).fit(selector.transform(z2),y)
        m,_,_=scores(model,selector.transform(zv2),yv);ablation.append({'experiment':tag,**m})
    pd.DataFrame(ablation).to_csv(out/'feature_ablation.csv',index=False)
    pd.DataFrame(ablation).plot(x='experiment',y='accuracy',kind='bar',legend=False);plt.ylim(0,1);plot_save(out,'11_feature_ablation')
    # Validation selection, then final untouched test evaluation
    candidates={**tuned,**ablation_models}
    candidate_scores={name:scores(pipe,rawv if name=='custom' else xv,yv)[0]['f1_macro'] for name,pipe in candidates.items()}
    winner=max(candidate_scores,key=candidate_scores.get);winner_pipe=candidates[winner]
    final_pipe=clone(winner_pipe)
    trainval_raw=raw+rawv;trainval_x=x+xv;trainval_y=np.r_[y,yv]
    final_pipe.fit(trainval_raw if winner=='custom' else trainval_x,trainval_y)
    test_input=rawt if winner=='custom' else xt
    final_metrics,pred,decision=scores(final_pipe,test_input,yt)
    pd.DataFrame([{'winner':winner,**final_metrics}]).to_csv(out/'final_test_metrics.csv',index=False)
    pd.DataFrame({'review':rawt,'actual':yt,'predicted':pred,'score':decision}).to_csv(out/'test_predictions.csv',index=False)
    (out/'classification_report.txt').write_text(classification_report(yt,pred,target_names=['negative','positive']),encoding='utf-8')
    joblib.dump(final_pipe,art/'best_pipeline.pkl',compress=3)
    if hasattr(final_pipe,'named_steps'):
        for step,obj in final_pipe.named_steps.items():joblib.dump(obj,art/('best_'+step+'.pkl'),compress=3)
    # confusion matrices, ROC and PR for all tuned models on validation; final test separate
    plt.figure();ConfusionMatrixDisplay.from_predictions(yt,pred,display_labels=['Negative','Positive'],cmap='Blues');plot_save(out,'12_final_confusion_matrix')
    plt.figure();fpr,tpr,_=roc_curve(yt,decision);plt.plot(fpr,tpr,label=f'{winner} (test)');plt.plot([0,1],[0,1],'--');plt.legend();plot_save(out,'13_final_roc')
    plt.figure();p,r,_=precision_recall_curve(yt,decision);plt.plot(r,p,label=winner);plt.legend();plot_save(out,'14_final_precision_recall')
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for name,pipe in tuned.items():
        _,yp,sc=scores(pipe,xv,yv);fpr,tpr,_=roc_curve(yv,sc);axs[0].plot(fpr,tpr,label=name)
        p,r,_=precision_recall_curve(yv,sc);axs[1].plot(r,p,label=name)
        ConfusionMatrixDisplay.from_predictions(yv,yp);plot_save(out,'confusion_'+name)
    axs[0].legend();axs[0].set_title('Validation ROC');axs[1].legend();axs[1].set_title('Validation PR');plot_save(out,'15_all_roc_pr')
    # model feature importance
    clf=final_pipe.named_steps.get('classifier');vectorizer=final_pipe.named_steps.get('vectorizer',final_pipe.named_steps.get('features'))
    if hasattr(clf,'coef_') and hasattr(vectorizer,'get_feature_names_out'):
        co=clf.coef_[0];names=vectorizer.get_feature_names_out();ix=np.argsort(abs(co))[-20:];plt.barh(names[ix],co[ix]);plot_save(out,'16_feature_importance')
    # tree importance from fitted random forest (on validation-selected representation)
    for key,(vv,model,reducer) in fitted.items():
        if 'RandomForest' in key and hasattr(model,'feature_importances_'):
            vec=make_vectorizer('bow',max_features=5000) if key.startswith('BoW_5k') else None
            imp=model.feature_importances_;ix=np.argsort(imp)[-20:];plt.barh([str(i) for i in ix],imp[ix]);plt.title('Random Forest feature indices');plot_save(out,'17_tree_importance');break
    # learning curve (training split only)
    lc=clone(winner_pipe);inputs=np.asarray(raw if winner=='custom' else x,dtype=object)
    sizes,tr,cv=learning_curve(lc,inputs,y,train_sizes=[.2,.4,.6,.8,1],cv=3,scoring='accuracy',n_jobs=1)
    plt.plot(sizes,tr.mean(axis=1),label='Train');plt.plot(sizes,cv.mean(axis=1),label='CV');plt.legend();plt.xlabel('Training examples');plt.ylabel('Accuracy');plot_save(out,'18_learning_curve')
    errors=pd.DataFrame({'review':rawt,'true':yt,'pred':pred,'score':decision});errors=errors[errors.true!=errors.pred]
    errors.head(20).to_csv(out/'20_misclassified_examples.csv',index=False)
    errors['length']=errors.review.str.len();plt.hist(errors['length'],bins=30);plt.title('Misclassified review lengths');plot_save(out,'19_error_analysis')
    # paired t-test: 5-fold scores on training, not final test
    cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
    s1=cross_val_score(Pipeline([('vectorizer',make_vectorizer('bow',max_features=5000)),('classifier',LogisticRegression(max_iter=2000))]),x,y,cv=cv,scoring='accuracy',n_jobs=1)
    s2=cross_val_score(Pipeline([('vectorizer',make_vectorizer('tfidf',max_features=5000)),('classifier',LogisticRegression(max_iter=2000))]),x,y,cv=cv,scoring='accuracy',n_jobs=1)
    stat,pval=ttest_rel(s1,s2);(out/'significance.json').write_text(json.dumps({'bow_folds':s1.tolist(),'tfidf_folds':s2.tolist(),'paired_t':float(stat),'p_value':float(pval)},indent=2))
    (out/'summary.json').write_text(json.dumps({'dataset_rows':len(df),'split_sizes':[len(a),len(v),len(t)],'winner':winner,'test_metrics':final_metrics},indent=2))
    report=f'''# Technical Report: IMDB Sentiment Analysis\n\n## Introduction and objective\nCompare text representations and supervised classifiers for binary movie review sentiment classification.\n\n## Methodology\nDeduplicate review texts, stratify into 60/20/20 train/validation/test partitions, clean text, preserve negation stopwords, optionally lemmatize with WordNet. Fit vocabularies and models only on training data. Compare BoW, TF-IDF, word n-grams and character n-grams.\n\n## Models and tuning\nCompare seven estimator variants from six classifier families. Run 5-fold randomized search for Naive Bayes, Logistic Regression and Linear SVM. Select using validation macro F1. Refit selected model on train plus validation before a single held-out test evaluation.\n\n## Results\nSelected model: **{winner}**. Test accuracy: **{final_metrics['accuracy']:.4f}**; macro F1: **{final_metrics['f1_macro']:.4f}**; ROC AUC: **{final_metrics['roc_auc']:.4f}**. Consult validation_comparison.csv, hyperparameter_tuning.csv, feature_ablation.csv, ngram_study.csv, and significance.json for quantitative comparisons.\n\n## Statistical significance\nPaired five-fold training CV t-test for BoW versus TF-IDF Logistic Regression: t={stat:.4f}, p={pval:.4g}. Shared folds are correlated; treat this as exploratory rather than conclusive independent-sample evidence.\n\n## Error analysis\nThe file 20_misclassified_examples.csv contains up to twenty held-out errors. Inspect negation, sarcasm, mixed sentiment, long reviews, rare vocabulary and rating mentions manually; do not claim a pattern without review.\n\n## Limitations and future work\nNo guarantee of 90% accuracy. Classical features miss some context, irony and domain shifts. Extend to pretrained embeddings, calibrated ensembles, LIME and cross-dataset tests when additional datasets and compute are available.\n'''
    (out/'technical_report.md').write_text(report,encoding='utf-8')
    try:
        from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from xml.sax.saxutils import escape
        styles=getSampleStyleSheet();story=[]
        for line in report.splitlines():
            if not line.strip():story.append(Spacer(1,8));continue
            st=styles['Heading1'] if line.startswith('# ') else styles['Heading2'] if line.startswith('## ') else styles['BodyText']
            story.append(Paragraph(escape(line.lstrip('# ').replace('**','')),st))
        SimpleDocTemplate(str(out/'technical_report.pdf'),pagesize=A4).build(story)
    except Exception as e:print('PDF report skipped:',e)
    # compact submission: code + notebook + short report + selected images, no dataset or models
    source=Path(__file__).resolve().parent
    allowed=['run_project.py','vectorization.py','models.py','feature_engineering.py','evaluation.py','requirements.txt','README.md','workflow.ipynb']
    with zipfile.ZipFile(root/'IMDB_Sentiment_Submission_UNDER_50MB.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in allowed:
            f=source/name
            if f.exists():archive.write(f,name)
        for name in ['technical_report.pdf','technical_report.md','final_test_metrics.csv','validation_comparison.csv','hyperparameter_tuning.csv','feature_ablation.csv','20_misclassified_examples.csv']:
            f=out/name
            if f.exists():archive.write(f,'results/'+name)
        for f in sorted(out.glob('*.png')):
            if f.stat().st_size<600000:archive.write(f,'results/'+f.name)
    size=(root/'IMDB_Sentiment_Submission_UNDER_50MB.zip').stat().st_size/1048576
    print('Finished. ZIP size:',round(size,2),'MB; test accuracy:',final_metrics['accuracy'])
    if size>=50:raise RuntimeError('ZIP exceeded 50MB')
if __name__=='__main__':main()

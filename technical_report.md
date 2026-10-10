# Technical Report: IMDB Sentiment Analysis

## Introduction and objective
Compare text representations and supervised classifiers for binary movie review sentiment classification.

## Methodology
Deduplicate review texts, stratify into 60/20/20 train/validation/test partitions, clean text, preserve negation stopwords, optionally lemmatize with WordNet. Fit vocabularies and models only on training data. Compare BoW, TF-IDF, word n-grams and character n-grams.

## Models and tuning
Compare seven estimator variants from six classifier families. Run 5-fold randomized search for Naive Bayes, Logistic Regression and Linear SVM. Select using validation macro F1. Refit selected model on train plus validation before a single held-out test evaluation.

## Results
Selected model: **LinearSVC**. Test accuracy: **0.8417**; macro F1: **0.8416**; ROC AUC: **0.9117**. Consult validation_comparison.csv, hyperparameter_tuning.csv, feature_ablation.csv, ngram_study.csv, and significance.json for quantitative comparisons.

## Statistical significance
Paired five-fold training CV t-test for BoW versus TF-IDF Logistic Regression: t=-3.1008, p=0.03619. Shared folds are correlated; treat this as exploratory rather than conclusive independent-sample evidence.

## Error analysis
The file 20_misclassified_examples.csv contains up to twenty held-out errors. Inspect negation, sarcasm, mixed sentiment, long reviews, rare vocabulary and rating mentions manually; do not claim a pattern without review.

## Limitations and future work
No guarantee of 90% accuracy. Classical features miss some context, irony and domain shifts. Extend to pretrained embeddings, calibrated ensembles, LIME and cross-dataset tests when additional datasets and compute are available.

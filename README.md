# Comparative Study of Text Vectorization Techniques and Machine Learning Classifiers for Sentiment Analysis

## AI-Based Sentiment Analysis of IMDB Movie Reviews

An end-to-end **Natural Language Processing (NLP) and Machine Learning project** that investigates how different text vectorization techniques influence sentiment classification performance on movie reviews.

The project compares **Bag-of-Words (BoW), TF-IDF, word-level N-grams, and character-level N-grams** using multiple machine learning classifiers. It also incorporates text preprocessing, hyperparameter optimization, feature engineering, feature selection, statistical analysis, model evaluation, and visualizations.

The primary objective is to develop an effective sentiment classification system capable of identifying whether a movie review expresses **positive or negative sentiment**.

---

## 1. Project Overview

Sentiment analysis, also known as opinion mining, is a Natural Language Processing technique used to identify the emotional orientation expressed in textual data.

With the increasing availability of user-generated reviews, automatically analyzing opinions has become essential for recommendation systems, customer feedback analysis, social media monitoring, and business intelligence.

This project conducts a comparative study of traditional machine learning approaches for sentiment analysis.

### Project Objectives

- Implement and compare multiple text representation techniques.
- Analyze the effectiveness of Bag-of-Words and TF-IDF.
- Investigate unigram, bigram, and trigram features.
- Explore character-level N-grams for robust text representation.
- Train and evaluate multiple machine learning classifiers.
- Optimize model hyperparameters using cross-validation.
- Extract additional sentiment-related and statistical features.
- Perform feature selection using statistical and model-based techniques.
- Evaluate model accuracy, precision, recall, F1-score, and ROC-AUC.
- Analyze incorrectly classified movie reviews.
- Compare training time and inference performance.
- Save trained models for future sentiment prediction.
- Generate visualizations, evaluation reports, and reproducible results.

---

## 2. Problem Statement

Textual movie reviews contain subjective opinions, complex language structures, and varying sentiment expressions.

Traditional machine learning algorithms cannot directly process raw text. Therefore, text must first be converted into numerical representations.

Different text vectorization techniques produce different feature spaces, which can significantly influence classifier performance.

The problem addressed in this project is:

**To systematically compare text vectorization techniques and machine learning classifiers for binary sentiment classification of IMDB movie reviews, and determine which combinations provide the most effective balance between classification accuracy, computational efficiency, and interpretability.**

The study also investigates whether feature engineering and feature selection improve classification performance.

---

## 3. Dataset Description

### Dataset: IMDB Movie Reviews Dataset

**Source:** Stanford Large Movie Review Dataset / Kaggle IMDB Dataset

Stanford Dataset:

https://ai.stanford.edu/~amaas/data/sentiment/

Kaggle Dataset:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

### Dataset Characteristics

| Attribute | Description |
|-----------|-------------|
| Dataset Name | IMDB Movie Reviews |
| Dataset Type | Text Classification |
| Total Reviews | Approximately 50,000 |
| Positive Reviews | Approximately 25,000 |
| Negative Reviews | Approximately 25,000 |
| Input Feature | Movie Review Text |
| Target Feature | Sentiment |
| Number of Classes | 2 |
| Classification Type | Binary Classification |
| Data Format | CSV |
| Language | English |

### Dataset Columns

| Column | Description |
|--------|-------------|
| review | Original movie review text |
| sentiment | Positive or negative sentiment label |

### Target Encoding

| Sentiment | Encoded Label |
|-----------|---------------|
| Negative | 0 |
| Positive | 1 |

### Dataset Distribution Visualization

The following graph displays the distribution of positive and negative movie reviews.

![IMDB Sentiment Class Distribution](graphs/01_class_distribution.png)

**Figure 1:** Class distribution of the IMDB movie review dataset.

A balanced dataset reduces the risk of classifiers becoming biased toward the majority class.

The implementation checks class distribution before model training and uses stratified sampling to preserve the relative class proportions.

---

## 4. Technologies and Libraries

### Programming Language

- Python 3.10 or later

### Development Environment

- Jupyter Notebook
- JupyterLab
- Visual Studio Code
- Anaconda

### Libraries

| Library | Purpose |
|---------|---------|
| NumPy | Numerical operations |
| Pandas | Dataset manipulation |
| Scikit-learn | Machine learning and evaluation |
| NLTK | Text preprocessing and lemmatization |
| SciPy | Sparse matrices and statistical testing |
| Matplotlib | Data visualization |
| Seaborn | Statistical plots and heatmaps |
| XGBoost | Gradient boosting classification |
| Joblib | Model serialization |
| Pickle | Python object serialization |
| PyYAML | Configuration and metadata |
| ReportLab | PDF report generation |
| JSON | Structured result storage |

---

## 5. Project Workflow

The complete project follows the workflow below:

```text
IMDB Movie Reviews Dataset
             |
             v
       Data Loading
             |
             v
    Data Quality Checking
             |
             v
      Text Preprocessing
             |
             v
  Train / Validation / Test
         60 / 20 / 20
             |
             v
     Text Vectorization
             |
    +--------+---------+
    |        |         |
    v        v         v
   BoW     TF-IDF    N-grams
                      |
                      v
               Word / Character
             |
             v
    Machine Learning Models
             |
             v
    Hyperparameter Tuning
             |
             v
      Feature Engineering
             |
             v
       Feature Selection
             |
             v
       Model Evaluation
             |
             v
      Comparative Analysis
             |
             v
       Best Model Selection
             |
             v
       Final Test Evaluation
             |
             v
     Saved Model Pipeline
             |
             v
      Sentiment Prediction
```

---

## 6. Data Preprocessing

Raw movie reviews contain HTML tags, punctuation, unnecessary symbols, stopwords, and inconsistent text formatting.

A preprocessing pipeline is applied to prepare the reviews for machine learning.

### 6.1 Text Cleaning

Text cleaning includes:

- Converting text to lowercase.
- Removing HTML tags.
- Removing URLs.
- Removing unwanted special characters.
- Removing unnecessary punctuation.
- Normalizing whitespace.

### 6.2 Tokenization

Tokenization divides text into individual words or tokens.

Example:

**Original Text:**

```text
This movie was absolutely amazing!
```

**Tokenized Text:**

```python
["this", "movie", "was", "absolutely", "amazing"]
```

### 6.3 Stopword Removal

Common words such as "the", "is", "and", and "was" may contribute limited information to sentiment classification.

However, sentiment-sensitive words such as:

```text
not, no, never, nor, very, but
```

are preserved because they can influence the meaning of a review.

For example:

```text
The movie was not good.
```

Removing "not" would change the sentiment expressed in the sentence.

### 6.4 Lemmatization

Lemmatization converts words into their dictionary or base forms.

Examples:

| Original Word | Lemmatized Word |
|---------------|-----------------|
| Movies | Movie |
| Cars | Car |
| Actors | Actor |
| Reviews | Review |

The implementation uses the WordNet lemmatizer from NLTK.

### 6.5 Dataset Splitting

The processed dataset is divided into:

| Dataset | Percentage |
|---------|------------|
| Training | 60% |
| Validation | 20% |
| Testing | 20% |

For a dataset containing exactly 50,000 usable reviews, the nominal split is:

- Training: 30,000 reviews
- Validation: 10,000 reviews
- Testing: 10,000 reviews

The actual numbers may differ after duplicate removal or when quick mode is enabled.

Stratified splitting preserves class proportions.

---

## 7. Text Vectorization Techniques

Machine learning algorithms require numerical inputs.

Text vectorization converts textual movie reviews into numerical feature vectors.

This project evaluates four major representation approaches.

### 7.1 Bag-of-Words (BoW)

Bag-of-Words represents text using the frequency of words in each document.

Example:

```text
Review 1: The movie was good
Review 2: The movie was bad
```

Possible vocabulary:

```text
[bad, good, movie, the, was]
```

Each review is converted into a vector containing word frequencies.

**Implementation:**

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.9
)

X_bow = vectorizer.fit_transform(X_train)
```

#### Parameters Evaluated

| Parameter | Values |
|-----------|--------|
| max_features | 1000, 5000, 10000 |
| min_df | 2, 5, 10 |
| max_df | 0.8, 0.9, 1.0 |
| binary | True, False |

#### Advantages

- Simple to implement.
- Computationally efficient.
- Easy to interpret.
- Works effectively with traditional classifiers.

#### Limitations

- Ignores word order.
- Produces sparse matrices.
- Does not capture semantic meaning.
- Common words may dominate feature counts.

---

### 7.2 TF-IDF Vectorization

TF-IDF stands for **Term Frequency–Inverse Document Frequency**.

It measures the importance of a word in a document relative to its frequency across the entire document collection.

The TF-IDF weighting concept is:

```text
TF-IDF(term, document) = TF(term, document) × IDF(term)
```

Terms appearing frequently in a particular document but less frequently across the corpus generally receive higher importance.

**Implementation:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_tfidf = vectorizer.fit_transform(X_train)
```

#### Parameters Evaluated

| Parameter | Values |
|-----------|--------|
| norm | l1, l2, None |
| use_idf | True, False |
| smooth_idf | True, False |
| sublinear_tf | True, False |

#### Advantages

- Reduces the influence of frequently occurring words.
- Highlights informative terms.
- Produces sparse numerical representations.
- Often works effectively with linear classifiers.

#### Limitations

- Does not directly understand context.
- Cannot fully interpret sarcasm.
- Vocabulary-dependent.
- Word-level TF-IDF cannot directly represent unseen words.

---

### 7.3 Word-Level N-grams

N-grams represent consecutive sequences of words.

#### Unigrams

Individual words:

```text
["very", "good", "movie"]
```

#### Bigrams

Pairs of consecutive words:

```text
["very good", "good movie"]
```

#### Trigrams

Sequences of three consecutive words:

```text
["very good movie"]
```

**Implementation:**

```python
CountVectorizer(ngram_range=(1, 1))
CountVectorizer(ngram_range=(2, 2))
CountVectorizer(ngram_range=(3, 3))
CountVectorizer(ngram_range=(1, 2))
CountVectorizer(ngram_range=(1, 3))
CountVectorizer(ngram_range=(2, 3))
```

#### Importance of N-grams

N-grams capture short-range word relationships that unigram representations cannot directly express.

For example:

```text
not good
very bad
highly recommended
waste of time
```

These expressions can provide valuable sentiment information.

However, higher-order N-grams may increase vocabulary size, sparsity, and computational requirements.

---

### 7.4 Character-Level N-grams

Character-level N-grams represent sequences of characters rather than complete words.

Example:

```text
amazing
```

Character bigrams:

```text
am, ma, az, zi, in, ng
```

**Implementation:**

```python
CountVectorizer(
    analyzer="char",
    ngram_range=(2, 4)
)
```

#### Advantages

- Captures subword patterns.
- Can provide robustness against spelling variations.
- Useful for unseen words and informal language.
- Does not require complete word matches.

#### Limitations

- Can generate large feature spaces.
- May introduce noisy features.
- Often requires additional computational resources.

---

## 8. Machine Learning Classifiers

The project implements seven classifiers across six major machine learning families.

### 8.1 Multinomial Naive Bayes

A probabilistic classifier based on Bayes' theorem.

It is commonly used for text classification and works with nonnegative word-count and TF-IDF features.

**Hyperparameter:**

```python
alpha
```

### 8.2 Bernoulli Naive Bayes

A probabilistic classifier designed for binary feature representations.

It considers whether a feature is present or absent.

**Hyperparameter:**

```python
alpha
```

### 8.3 Logistic Regression

A linear classification algorithm that estimates the probability of a binary outcome.

It is commonly effective for high-dimensional sparse text features.

**Hyperparameters:**

```python
C
penalty
class_weight
```

### 8.4 Support Vector Machine

Linear SVM identifies a separating decision boundary between positive and negative reviews.

It is suitable for high-dimensional sparse feature spaces.

**Hyperparameters:**

```python
C
max_iter
class_weight
```

### 8.5 Random Forest

Random Forest combines multiple decision trees to produce classification predictions.

It can capture nonlinear relationships between features.

**Hyperparameters:**

```python
n_estimators
max_depth
min_samples_split
```

### 8.6 XGBoost

XGBoost is a gradient boosting algorithm that builds trees sequentially to improve prediction performance.

**Hyperparameters:**

```python
learning_rate
n_estimators
max_depth
```

### 8.7 K-Nearest Neighbors

KNN predicts sentiment based on the labels of similar training reviews.

Similarity can be measured using cosine distance.

**Hyperparameters:**

```python
n_neighbors
metric
```

KNN inference can become computationally expensive for large text datasets.

---

## 9. Hyperparameter Optimization

Hyperparameter optimization is performed using:

- RandomizedSearchCV
- StratifiedKFold
- Five-fold cross-validation

The primary tuned classifiers are:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear SVM

Example:

```python
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=parameters,
    n_iter=12,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)
```

The best parameters and cross-validation results are saved for comparison.

---

## 10. Feature Engineering

Feature engineering introduces additional information that may not be fully captured by text vectorization.

### 10.1 Sentiment Lexicon Features

The implementation uses predefined positive and negative sentiment words.

Positive examples:

```text
amazing, excellent, wonderful, fantastic, brilliant
```

Negative examples:

```text
terrible, awful, boring, horrible, disappointing
```

Extracted features include:

- Positive word count
- Negative word count
- Sentiment polarity ratio

### 10.2 Statistical Features

| Feature | Description |
|---------|-------------|
| Character Count | Number of characters |
| Word Count | Number of words |
| Average Word Length | Mean length of words |
| Sentence Count | Number of detected sentences |
| Exclamation Count | Number of exclamation marks |
| Question Count | Number of question marks |
| Capital Ratio | Proportion of uppercase letters |

### 10.3 Domain-Specific Features

Additional features include:

- Presence of "amazing"
- Presence of "terrible"
- Presence of "worst"
- Movie rating mentions

### 10.4 Combining Features

Text vectorization features are combined with custom numerical features using sparse matrix concatenation.

```python
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

custom_scaled = scaler.fit_transform(custom_features)

combined_features = hstack([
    tfidf_features,
    custom_scaled
])
```

The project compares the performance of TF-IDF alone against TF-IDF combined with engineered features.

---

## 11. Feature Selection

Feature selection identifies useful features and removes less informative variables.

Three techniques are evaluated.

### Chi-Square Feature Selection

Measures statistical association between nonnegative text features and sentiment classes.

### Mutual Information

Estimates the dependency between input features and target labels.

### L1-Based Feature Selection

Uses L1-regularized Logistic Regression to identify features with nonzero coefficients.

### Objectives

- Reduce feature dimensionality.
- Investigate model efficiency.
- Reduce unnecessary features.
- Compare classification performance.
- Examine the effect of dimensionality reduction.

---

## 12. Model Evaluation

The project evaluates models using multiple classification metrics.

### Accuracy

The proportion of correctly classified reviews.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

Measures how many predicted positive reviews are actually positive.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual positive reviews are correctly identified.

```text
Recall = TP / (TP + FN)
```

### F1-Score

The harmonic mean of precision and recall.

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

### ROC-AUC

Measures how effectively the classifier distinguishes positive and negative reviews across classification thresholds.

### Additional Evaluation

- Macro Precision
- Macro Recall
- Macro F1
- Weighted Precision
- Weighted Recall
- Weighted F1
- Confusion Matrix
- Classification Report
- Training Time
- Inference Time

---

## 13. Comparative Analysis

The project generates comparison tables covering:

| Comparison | Description |
|------------|-------------|
| BoW vs TF-IDF | Influence of weighting strategy |
| Unigrams vs N-grams | Influence of word sequence length |
| Word vs Character N-grams | Word and subword representations |
| Classifier Comparison | Accuracy and F1 across algorithms |
| Training Time | Computational efficiency |
| Inference Time | Prediction speed |
| Feature Engineering | Baseline vs engineered features |
| Feature Selection | Performance with reduced features |
| Hyperparameter Tuning | Optimized classifier configurations |

### Research Questions

The experiments investigate:

1. Which vectorization technique performs best?
2. Which classifier achieves the highest accuracy?
3. How does vocabulary size influence performance?
4. Do bigrams and trigrams improve sentiment classification?
5. Which classifier provides the best accuracy-speed trade-off?
6. Does feature engineering improve accuracy?
7. Which custom features contribute most?
8. Does feature selection reduce computation without harming accuracy?
9. Which model is most sensitive to hyperparameter tuning?
10. What types of reviews are most frequently misclassified?

Actual answers are determined from the generated experiment results.

---

## 14. Visualizations

The implementation generates multiple graphs to support the comparative study.

### Dataset Analysis

![Class Distribution](graphs/01_class_distribution.png)

### Vocabulary Size

![Vocabulary Size](graphs/03_vocabulary_size.png)

### Feature Sparsity

![Feature Sparsity](graphs/04_bow_sparsity_heatmap.png)

### Top TF-IDF Features

![TF-IDF Features](graphs/06_top_tfidf_features.png)

### N-gram Performance

![N-gram Accuracy](graphs/09_ngram_accuracy.png)

### Classifier Accuracy

![Model Accuracy](graphs/12_model_accuracy_comparison.png)

### Classifier F1-Score

![Model F1](graphs/13_model_f1_comparison.png)

### ROC Curves

![ROC Curves](graphs/14_roc_curves.png)

### Precision-Recall Curves

![Precision Recall](graphs/15_precision_recall_curves.png)

### Training Time

![Training Time](graphs/16_training_time_comparison.png)

### Feature Engineering

![Feature Engineering](graphs/18_feature_engineering_comparison.png)

### Feature Selection

![Feature Selection](graphs/19_feature_selection_comparison.png)

### Learning Curve

![Learning Curve](graphs/23_learning_curve.png)

### Final Confusion Matrix

![Final Confusion Matrix](graphs/24_final_test_confusion_matrix.png)

### Vectorization Comparison

![Vectorization Comparison](graphs/26_vectorization_accuracy_comparison.png)

All figures are generated during execution and saved in the graphs directory.

---

## 15. Statistical Significance Testing

A paired t-test is performed to compare the cross-validation accuracy of Logistic Regression and Linear SVM.

### Hypotheses

**Null Hypothesis (H0):**

There is no difference in mean accuracy between the two classifiers.

**Alternative Hypothesis (H1):**

The classifiers have different mean accuracies.

### Significance Level

```text
alpha = 0.05
```

### Interpretation

- p-value below 0.05 indicates a difference under the assumptions of the test.
- p-value at or above 0.05 indicates insufficient evidence to reject the null hypothesis.

Because cross-validation fold results are correlated, this paired t-test is treated as exploratory rather than definitive statistical proof.

Results are saved in:

```text
results/statistical_significance.csv
results/cross_validation_fold_results.csv
```

---

## 16. Error Analysis

The project identifies incorrectly classified reviews and saves detailed prediction results.

The analysis investigates potential failure patterns such as:

- Negation
- Sarcasm
- Mixed positive and negative opinions
- Ambiguous sentiment
- Complex sentence structure
- Unusual review length
- Contrast expressions
- Rating mentions

Twenty misclassified examples are extracted for manual inspection.

### Error Analysis Outputs

```text
results/all_misclassified_reviews.csv
results/20_misclassified_examples.csv
results/error_summary.csv
```

These outputs help identify weaknesses in the selected classifier.

---

## 17. Project Directory Structure

```text
Text Representation & Machine Learning Classification/
|
|-- IMDB Dataset.csv
|-- README.md
|-- IMDB_All_In_One_Cell.ipynb
|-- IMDB_All_In_One_Cell.py
|
|-- models/
|   |-- bow_vectorizer.pkl
|   |-- tfidf_vectorizer.pkl
|   |-- character_vectorizer.pkl
|   |-- MultinomialNB.pkl
|   |-- BernoulliNB.pkl
|   |-- LogisticRegression.pkl
|   |-- LinearSVM.pkl
|   |-- RandomForest.pkl
|   |-- XGBoost.pkl
|   |-- KNN.pkl
|   |-- tuned_MultinomialNB.pkl
|   |-- tuned_LogisticRegression.pkl
|   |-- tuned_LinearSVM.pkl
|   |-- custom_feature_scaler.pkl
|   |-- engineered_logistic_regression.pkl
|   |-- best_sentiment_pipeline.pkl
|
|-- graphs/
|   |-- 01_class_distribution.png
|   |-- 02_split_distribution.png
|   |-- 03_vocabulary_size.png
|   |-- 04_bow_sparsity_heatmap.png
|   |-- 05_top_bow_features.png
|   |-- 06_top_tfidf_features.png
|   |-- 07_tfidf_distribution.png
|   |-- 08_sample_tfidf_scores.png
|   |-- 09_ngram_accuracy.png
|   |-- 10_ngram_training_time.png
|   |-- 12_model_accuracy_comparison.png
|   |-- 13_model_f1_comparison.png
|   |-- 14_roc_curves.png
|   |-- 15_precision_recall_curves.png
|   |-- 16_training_time_comparison.png
|   |-- 17_inference_time_comparison.png
|   |-- 18_feature_engineering_comparison.png
|   |-- 19_feature_selection_comparison.png
|   |-- 20_random_forest_feature_importance.png
|   |-- 21_xgboost_feature_importance.png
|   |-- 22_custom_feature_importance.png
|   |-- 23_learning_curve.png
|   |-- 24_final_test_confusion_matrix.png
|   |-- 25_error_length_distribution.png
|   |-- 26_vectorization_accuracy_comparison.png
|   |-- 27_vectorization_time_comparison.png
|   |-- confusion_matrix_*.png
|
|-- results/
|   |-- preprocessed_imdb.csv
|   |-- train_preprocessed.csv
|   |-- validation_preprocessed.csv
|   |-- test_preprocessed.csv
|   |-- bow_parameter_experiments.csv
|   |-- tfidf_parameter_experiments.csv
|   |-- ngram_experiments.csv
|   |-- model_comparison.csv
|   |-- vectorization_model_comparison.csv
|   |-- hyperparameter_tuning.csv
|   |-- feature_engineering_comparison.csv
|   |-- feature_selection_results.csv
|   |-- custom_feature_importance.csv
|   |-- learning_curves.csv
|   |-- statistical_significance.csv
|   |-- final_model_selection.csv
|   |-- final_test_metrics.csv
|   |-- test_predictions.csv
|   |-- all_misclassified_reviews.csv
|   |-- 20_misclassified_examples.csv
|   |-- project_summary.json
|   |-- project_summary.yaml
|   |-- technical_report.md
|   |-- technical_report.pdf
|
|-- submission/
|   |-- environment.json
|
|-- IMDB_Sentiment_Submission_UNDER_50MB.zip
```

Some files are generated only after their corresponding experiments complete.

---

## 18. Installation Instructions

### Step 1: Install Python

Install Python 3.10 or later.

Official website:

https://www.python.org/downloads/

### Step 2: Install Jupyter Notebook

Open Command Prompt or Anaconda Prompt.

```bash
pip install notebook
```

### Step 3: Install Required Libraries

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn nltk joblib pyyaml reportlab xgboost ipython nbformat
```

### Step 4: Download the Dataset

Download the IMDB dataset and place it in the project directory.

Expected filename:

```text
IMDB Dataset.csv
```

### Step 5: Open Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
IMDB_All_In_One_Cell.ipynb
```

### Step 6: Configure Dataset Path

```python
from pathlib import Path

ROOT = Path(
    r"C:\Users\sagni\Downloads\Text Representation & Machine Learning Classification"
)

DATASET = ROOT / "IMDB Dataset.csv"
```

### Step 7: Run the Notebook

Execute the notebook cell.

The program automatically creates output folders, trains models, evaluates performance, and saves generated results.

---

## 19. Execution Modes

The project supports two execution modes.

### Quick Mode

```python
QUICK_MODE = True
```

This mode uses a smaller sample of the dataset.

Recommended for:

- Initial testing
- Debugging
- Checking library compatibility
- Verifying output generation

### Full Mode

```python
QUICK_MODE = False
```

This mode uses the full available dataset after preprocessing and duplicate removal.

Recommended for:

- Final experimentation
- Comparative analysis
- Final model evaluation
- Academic reporting

**Note:** Some computationally expensive classifiers may still use limited training subsets for efficiency.

---

## 20. Model Saving and Deployment

The project saves trained classifiers and vectorizers using Joblib.

### Best Model Pipeline

```text
models/best_sentiment_pipeline.pkl
```

This pipeline includes the text vectorizer and selected classifier.

### Loading the Model

```python
import joblib

model = joblib.load(
    "models/best_sentiment_pipeline.pkl"
)
```

### Example Prediction

```python
reviews = [
    "This movie was absolutely amazing!",
    "The acting was terrible and the story was boring."
]

# Apply the same preprocessing used during training.
cleaned_reviews = [
    clean_text(review) for review in reviews
]

predictions = model.predict(cleaned_reviews)

for review, prediction in zip(reviews, predictions):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(review, "=>", sentiment)
```

**Important:** The saved pipeline expects text that has undergone the same cleaning and lemmatization used during training. The preprocessing function must be available at prediction time.

---

## 21. Results and Performance Targets

The project evaluates model performance against the following target levels.

| Performance Level | Test Accuracy |
|-------------------|---------------|
| Minimum Target | 85% |
| Good | 88–90% |
| Excellent | Above 90% |

These values are project targets, not guaranteed or precomputed results.

### Final Results

After execution, actual performance metrics are available in:

```text
results/final_test_metrics.csv
```

The comparison between classifiers is saved in:

```text
results/model_comparison.csv
```

The vectorization comparison is saved in:

```text
results/vectorization_model_comparison.csv
```

No accuracy or F1-score is claimed until the corresponding experiment has been executed.

---

## 22. Computational Efficiency

Different machine learning models have different computational requirements.

### Relatively Efficient Models

- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- Logistic Regression
- Linear SVM

### More Computationally Expensive Models

- Random Forest
- XGBoost
- K-Nearest Neighbors

Training and inference times are recorded to study the relationship between model complexity and prediction performance.

The comparison helps identify models suitable for environments with limited computational resources.

---

## 23. Submission ZIP Optimization

The project generates a submission archive with a target size below 50 MB.

### Included Files

- Source notebook, when explicitly included in the archive
- Technical report
- Evaluation results
- Comparison tables
- Selected graphs
- Configuration metadata

### Excluded Files

- Original IMDB dataset
- Large trained model files
- Large intermediate datasets
- Python virtual environments
- Cache files
- Unnecessary temporary outputs

### ZIP Output

```text
IMDB_Sentiment_Submission_UNDER_50MB.zip
```

The trained models remain available locally even when excluded from the submission archive.

Before submission, verify that the notebook source is included in the ZIP and that the final archive satisfies the size restriction.

---

## 24. Expected Learning Outcomes

After completing this project, learners will understand:

1. How to preprocess large text datasets.
2. How to convert text into numerical representations.
3. How Bag-of-Words differs from TF-IDF.
4. How word-level N-grams capture local context.
5. How character N-grams represent subword patterns.
6. How to train traditional machine learning classifiers.
7. How to optimize hyperparameters using cross-validation.
8. How to evaluate binary classification performance.
9. How to perform custom feature engineering.
10. How to implement feature selection.
11. How to compare computational cost and accuracy.
12. How to perform statistical model comparisons.
13. How to identify common classification errors.
14. How to save and reuse trained machine learning pipelines.
15. How to document reproducible NLP experiments.

---

## 25. Applications

The techniques developed in this project can be applied to:

### Movie Review Classification

Automatically categorize movie reviews into positive and negative sentiment.

### Product Review Analysis

Analyze customer feedback on e-commerce platforms.

### Social Media Sentiment Analysis

Identify opinions expressed in social media posts.

### Customer Feedback Monitoring

Help organizations understand customer satisfaction.

### Brand Reputation Analysis

Measure positive and negative public opinions about brands.

### Recommendation Systems

Use sentiment information to support personalized recommendations.

### Market Research

Analyze large collections of customer opinions and review data.

---

## 26. Limitations

Despite its practical usefulness, the project has several limitations.

- Traditional vectorization techniques cannot fully understand contextual meaning.
- Sarcasm and irony may lead to incorrect predictions.
- Mixed-sentiment reviews are difficult to classify.
- Vocabulary size affects memory usage and training time.
- Character N-grams may introduce noisy features.
- Tree-based models can be computationally expensive on sparse text.
- KNN inference becomes slower as training data increases.
- The results may not generalize directly to other review domains.
- Hyperparameter tuning does not guarantee optimal performance.
- Accuracy depends on preprocessing, dataset quality, and model configuration.

---

## 27. Future Enhancements

Potential improvements include:

### Deep Learning

Implement LSTM, BiLSTM, or CNN-based sentiment classifiers.

### Transformer Models

Compare traditional classifiers against BERT, DistilBERT, or RoBERTa.

### Ensemble Learning

Combine multiple classifiers using voting or stacking.

### Explainable AI

Use LIME or SHAP to explain individual predictions.

### Cross-Dataset Evaluation

Train on IMDB reviews and evaluate on Amazon product reviews.

### Web Application

Develop a Streamlit or Flask application for real-time sentiment classification.

### Advanced Hyperparameter Optimization

Explore Bayesian optimization and Optuna.

### Better Negation Handling

Improve text preprocessing to preserve more complex sentiment relationships.

---

## 28. Conclusion

This project presents a systematic comparative study of text vectorization techniques and machine learning classifiers for sentiment analysis using IMDB movie reviews.

By evaluating Bag-of-Words, TF-IDF, word N-grams, and character N-grams, the study investigates how different numerical text representations influence classification accuracy, feature dimensionality, and computational cost.

Multiple machine learning classifiers are trained and compared using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and execution time.

Feature engineering and feature selection are additionally evaluated to determine whether they improve classification performance.

The final model is selected using validation performance and evaluated on a separate test set.

The project demonstrates a reproducible approach to classical NLP experimentation and provides a foundation for future improvements using deep learning, transformer models, and explainable AI.

---

## 29. References

1. Maas, A. L., et al. (2011). Learning Word Vectors for Sentiment Analysis. ACL.
   https://ai.stanford.edu/~amaas/data/sentiment/

2. Scikit-learn Documentation.
   https://scikit-learn.org/stable/

3. NLTK Documentation.
   https://www.nltk.org/

4. XGBoost Documentation.
   https://xgboost.readthedocs.io/

5. Pandas Documentation.
   https://pandas.pydata.org/

6. NumPy Documentation.
   https://numpy.org/

7. Matplotlib Documentation.
   https://matplotlib.org/

8. SciPy Documentation.
   https://scipy.org/

---

## 30. Project Status

**Implementation:** Python and Scikit-learn

**Dataset:** IMDB Movie Reviews

**Task:** Binary Sentiment Classification

**Text Representations:** BoW, TF-IDF, Word N-grams, Character N-grams

**Machine Learning:** Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost, KNN

**Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC

**Feature Engineering:** Statistical, Lexicon-Based, Domain-Specific

**Output:** CSV, PNG, PKL, JSON, YAML, PDF, Markdown, ZIP

**Reproducibility:** Random Seed 42

**Project Type:** Natural Language Processing / Machine Learning / Comparative Analysis

---

## License

This project is intended for educational, academic, and research purposes.

Dataset usage remains subject to the terms of its original provider.

---

**Comparative Study of Text Vectorization Techniques and Machine Learning Classifiers for Sentiment Analysis**

*An end-to-end NLP project for sentiment classification, feature engineering, model comparison, and performance analysis.*

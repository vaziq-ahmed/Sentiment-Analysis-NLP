# Sentiment Analysis — Product Review Classifier
 
An end-to-end NLP project that classifies product reviews into **Positive**,
**Neutral**, or **Negative** sentiment, using classical ML (TF-IDF +
Logistic Regression / Linear SVM), wrapped in an interactive **Streamlit**
web app.
 
---
 
## Table of contents
 
- [Overview](#overview)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Pipeline](#pipeline)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Preprocessing](#2-preprocessing)
  - [3. Feature engineering & train/test split](#3-feature-engineering--traintest-split)
  - [4. Model building](#4-model-building)
  - [5. Hyperparameter tuning](#5-hyperparameter-tuning)
  - [6. Saving the final model](#6-saving-the-final-model)
- [Streamlit app](#streamlit-app)
- [Setup & installation](#setup--installation)
- [Running the app](#running-the-app)
- [Deployment (Streamlit Community Cloud)](#deployment-streamlit-community-cloud)
- [Tech stack](#tech-stack)
- [Future improvements](#future-improvements)
---
 
## Overview
 
This project takes raw e-commerce product review data (title, body, star
rating) and builds a machine learning model that predicts the **sentiment**
of a review directly from its text — without needing the star rating.
 
The workflow covers the full data science lifecycle:
 
```
Raw data → EDA → Label creation → Text cleaning → Feature engineering
        → Model training → Hyperparameter tuning → Saved model → Streamlit app
```
 
---
 
## Dataset
 
| Property        | Details                                  |
|------------------|-------------------------------------------|
| Source file      | `Sentiment_Analysis_Dataset.xlsx`         |
| Rows             | 1,440                                     |
| Columns          | `title`, `rating`, `body`                 |
| Missing values   | None                                       |
| Language         | Mostly English, some Hindi                |
 
**Rating distribution:**
 
| Rating | Count | Mapped sentiment |
|--------|-------|-------------------|
| 1 ⭐    | 386   | Negative          |
| 2 ⭐    | 126   | Negative          |
| 3 ⭐    | 199   | Neutral           |
| 4 ⭐    | 310   | Positive          |
| 5 ⭐    | 419   | Positive          |
 
Star ratings are converted into a 3-class sentiment label
(`Negative` / `Neutral` / `Positive`), which becomes the target variable.
 
---
 
## Project structure
 
```
SentimentAnalysis_NLP/
│
├── app.py                  # Streamlit app entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation (this file)
│
├── sentiment_model.pkl     # Trained classifier (saved with joblib)
├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
├── label_encoder.pkl       # Label encoder for sentiment classes
│
├── notebook/
│   └── sentiment_analysis.ipynb   # Full EDA -> model building notebook
│
└── assets/
    ├── __init__.py
    ├── preprocessing.py     # Text cleaning + prediction helper functions
    └── style.css            # Custom front-end styling for the app
```
 
---
 
## Pipeline
 
### 1. Exploratory Data Analysis (EDA)
 
- Checked shape, column types, missing values, and duplicates.
- Analyzed **rating distribution** (count plot + pie chart) — found a
  slight positive skew (729 reviews rated 4–5 vs 512 rated 1–2).
- Computed **review length statistics** (character count, word count) and
  compared distributions across ratings using box plots and histograms.
- Generated **word clouds** for negative, neutral, and positive review
  groups to spot recurring themes.
- Extracted **top frequent words** per sentiment group using a simple
  frequency counter with stopword filtering.
### 2. Preprocessing
 
- **Label creation**: mapped star ratings → sentiment labels
  (1–2 → Negative, 3 → Neutral, 4–5 → Positive).
- **Text combination**: merged `title` + `body` into a single `full_text`
  field for richer context.
- **Cleaning**: lowercased text, removed URLs, HTML tags, digits,
  punctuation, and extra whitespace.
- **Tokenization & stopword removal**: tokenized with NLTK, removed common
  English stopwords — **except negation words** (`not`, `no`, `nor`,
  `don't`, etc.), since these are critical for sentiment meaning.
- **Lemmatization**: reduced tokens to their root form using
  `WordNetLemmatizer`.
- Dropped any rows that became empty after cleaning and exported the
  cleaned dataset.
### 3. Feature engineering & train/test split
 
- **Label encoding**: converted `Negative` / `Neutral` / `Positive` into
  numeric labels using `LabelEncoder`.
- **Stratified train/test split** (80/20) to preserve class proportions in
  both sets.
- **TF-IDF vectorization**: converted cleaned text into numerical features
  using unigrams + bigrams (`ngram_range=(1,2)`), capped at 5,000 features,
  with `min_df=2` and `max_df=0.9`.
- **Class weights**: computed balanced class weights to address the
  imbalance between sentiment classes (Neutral and Negative are
  under-represented compared to Positive).
### 4. Model building
 
Trained and compared four baseline models on the TF-IDF features:
 
| Model               | Why it was included                                  |
|---------------------|--------------------------------------------------------|
| Logistic Regression | Strong, fast baseline for sparse text data             |
| Multinomial Naive Bayes | Classic text-classification algorithm, very fast   |
| Linear SVM          | Often top performer on high-dimensional sparse data    |
| Random Forest       | Included for comparison/contrast with linear models    |
 
Models were evaluated using **accuracy** and **F1-macro** (macro F1 is more
informative here due to class imbalance), with a confusion matrix for the
best-performing model.
 
### 5. Hyperparameter tuning
 
- **Logistic Regression**: tuned `C` (regularization strength), `penalty`,
  and `solver` via `GridSearchCV` with 5-fold cross-validation, scored on
  F1-macro.
- **Linear SVM**: tuned `C` and `loss` (`hinge` vs `squared_hinge`) with the
  same cross-validation setup.
- Compared tuned models against baselines on the held-out test set.
### 6. Saving the final model
 
- Selected the best-performing tuned model based on test F1-macro.
- Saved three artifacts with `joblib`:
  - `sentiment_model.pkl` — the trained classifier
  - `tfidf_vectorizer.pkl` — the fitted TF-IDF vectorizer
  - `label_encoder.pkl` — the label encoder for sentiment classes
- Wrote a reusable `predict_sentiment(text)` function that runs the full
  preprocessing pipeline and returns a sentiment label for any new review.
---
 
## Streamlit app
 
The `app.py` file provides an interactive web interface built on top of the
saved model artifacts.
 
**Features:**
 
- **Single review tab** — enter or paste a review and get an instant
  sentiment prediction, with a confidence breakdown chart across all three
  classes.
- **Batch (CSV) tab** — upload a CSV of reviews, choose the text column,
  run predictions on all rows, view a sentiment distribution pie chart, and
  download the results as CSV.
- **History tab** — view all predictions made during the current session
  with a summary bar chart.
- **Sidebar** — quick example reviews (positive / negative / neutral) to
  try the model instantly, plus a "clear history" option.
**Code organization:**
 
- `assets/preprocessing.py` — contains `clean_text()`, `preprocess()`,
  `load_artifacts()`, and `get_prediction()`. These mirror the exact
  preprocessing steps used during training, so predictions stay consistent.
- `assets/style.css` — custom styling for the hero banner and the
  color-coded sentiment result box (green/red/amber for
  positive/negative/neutral).
- `app.py` — wires everything together: page config, sidebar, KPI metrics,
  tabs, and the prediction logic.
---
 
## Setup & installation
 
1. Clone or download this project folder.
2. (Recommended) create a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
```
 
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
 
4. Make sure the following files exist in the project root (generated from
   the notebook):
   - `sentiment_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `label_encoder.pkl`
   If your filenames differ, update the paths in `get_artifacts()` inside
   `app.py`.
---
 
## Running the app
 
From the project root:
 
```bash
streamlit run app.py
```
 
The app will open in your browser at `http://localhost:8501`.
 
---
 
## Deployment (Streamlit Community Cloud)
 
1. Push the entire project folder — including the `.pkl` files — to a
   GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, select your repository and branch, and set the main
   file path to `app.py`.
4. Click **Deploy**. Your app will be live at a URL like:
```
   https://your-app-name.streamlit.app
```
 
> **Note on versions**: if you hit `ModuleNotFoundError` or unpickling
> errors after deployment, it usually means the `numpy` / `scikit-learn` /
> `joblib` versions on the deployment environment differ from the ones used
> to save the `.pkl` files. Pin compatible versions in `requirements.txt`,
> or re-save the artifacts using the deployment environment's versions.
 
---
 
## Tech stack
 
- **Language**: Python 3.11
- **Data handling**: pandas, numpy, openpyxl
- **NLP**: NLTK (tokenization, stopwords, lemmatization), WordCloud
- **Machine learning**: scikit-learn (TF-IDF, Logistic Regression,
  Naive Bayes, Linear SVM, Random Forest, GridSearchCV)
- **Visualization**: matplotlib, seaborn, plotly
- **Web app**: Streamlit
- **Model persistence**: joblib
---
 
## Future improvements
 
- Try word embeddings (Word2Vec / GloVe) or transformer-based models
  (e.g. fine-tuned BERT/DistilBERT) and compare against TF-IDF + classical
  ML.
- Perform deeper error analysis on the Neutral class, which is the
  smallest and hardest to classify.
- Add multilingual support for the Hindi-language reviews present in the
  dataset.
- Add model explainability (e.g. LIME/SHAP) to the Streamlit app to show
  which words influenced a prediction.
 
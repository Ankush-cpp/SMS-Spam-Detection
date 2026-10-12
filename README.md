# 📱 SMS Spam Detection System

A complete, beginner-friendly Machine Learning project that classifies SMS messages as **Spam** or **Not Spam (Ham)** using a TF-IDF + Multinomial Naive Bayes pipeline, served through an interactive Streamlit web application.

---

## 🌟 Features

- **Real dataset** — trained on the publicly available [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,572 rows in the distributed file; 5,169 unique messages after deduplication)
- **Text preprocessing** — lowercasing, URL/number normalisation, punctuation removal
- **TF-IDF vectorisation** with unigram + bigram features (10,000 max)
- **Multinomial Naive Bayes** classifier (α = 0.1)
- **Stratified 80/20 train-test split** for unbiased evaluation
- **Model persistence** — trained pipeline saved to `model/spam_pipeline.joblib`
- **Interactive Streamlit UI** with:
  - SMS text area + Predict button
  - Clear Spam / Not Spam result with confidence score
  - Probability bar chart per prediction
  - Model evaluation tab (accuracy, precision, recall, F1, confusion matrix)
  - Dataset explorer with class distribution and message-length plots
  - Built-in sample messages for quick testing
- **Proper error handling** and input validation throughout

---

## 🗂️ Project Structure

```
SMS-Spam-Detection/
├── app.py               ← Streamlit web application
├── train_model.py       ← Dataset download, training, and evaluation script
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
├── data/
│   └── SMSSpamCollection   ← Downloaded automatically (tab-separated)
├── model/
│   ├── spam_pipeline.joblib  ← Trained TF-IDF + NB pipeline
│   └── metrics.json          ← Evaluation metrics (JSON)
└── assets/
    ├── confusion_matrix.png  ← Confusion matrix heatmap
    └── metrics_bar.png       ← Metrics bar chart
```

---

## 📦 Dataset

| Property | Value |
|---|---|
| Name | SMS Spam Collection |
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
| Published size | 5,574 SMS messages (4,827 ham · 747 spam) |
| Rows in distributed file | 5,572 |
| Unique messages (after deduplication) | 5,169 (653 spam · 4,516 ham) |
| Duplicates removed | 403 exact duplicate rows |
| Used for training / evaluation | 5,169 messages — 80/20 stratified split |
| Format | Tab-separated: `label\tmessage` |

> **Note on counts:** The UCI-published figure is 5,574. The distributed archive file
> contains 5,572 rows (a known minor discrepancy in the repository). After removing
> 403 exact duplicate rows the working dataset is **5,169 unique messages**.
> All model metrics (accuracy, precision, recall, F1) are computed on the 20 %
> held-out split of the **5,169** deduplicated messages (1,034 test rows).

The dataset is **downloaded automatically** when you run `train_model.py`.
If the automatic download fails (network issues), visit the URL above, download the ZIP, and extract `SMSSpamCollection` into the `data/` folder.

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9 or later
- `pip` package manager

### 1. Clone / navigate to the project folder

```bash
cd SMS-Spam-Detection
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

This will:
- Download the SMS Spam Collection dataset into `data/`
- Train the TF-IDF + Multinomial Naive Bayes pipeline
- Print accuracy, precision, recall, and F1 score to the terminal
- Save the trained model to `model/spam_pipeline.joblib`
- Save evaluation charts to `assets/`

### 5. Launch the web app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 🖥️ Usage

1. Go to the **🔍 Predict** tab.
2. Type or paste any SMS message into the text area (or click a sample on the right).
3. Click **Predict**.
4. The app displays:
   - **SPAM** (red) or **NOT SPAM** (green) verdict
   - Confidence percentages for both classes
   - A horizontal probability bar chart

Explore the **📊 Model Evaluation** tab to see confusion matrix, metric scores, and pipeline details.  
Browse the **📂 Dataset Explorer** tab to view class distributions and sample messages.

---

## 🔬 Methodology

### 1. Data Loading & Cleaning
- Read the tab-separated file with `pandas` (5,572 rows loaded from file)
- Drop exact duplicate rows and null rows → **5,169 unique messages**
- Encode labels: `ham → 0`, `spam → 1`

### 2. Text Preprocessing (inside TF-IDF)
```
lowercase → replace URLs with "url" → replace digits with "num"
→ remove punctuation → collapse whitespace
```

### 3. Feature Extraction — TF-IDF
- **Term Frequency–Inverse Document Frequency** converts text to numerical feature vectors
- Captures both single-word and two-word patterns (bigrams)
- `sublinear_tf=True` dampens very frequent terms
- Vocabulary capped at 10,000 features

### 4. Classification — Multinomial Naive Bayes
- A probabilistic classifier ideal for discrete (count-like) features
- Assumes feature independence given the class (Naive assumption)
- Smoothing α = 0.1 avoids zero probabilities for unseen words

### 5. Evaluation
- Stratified 80/20 train-test split on the **5,169** deduplicated messages (random state 42)
- Train: 4,135 messages · Test: 1,034 messages (131 spam, 903 ham)
- Metrics: **Accuracy, Precision, Recall, F1-score** (binary, positive = Spam)
- Confusion matrix visualised as a heatmap

---

## 📊 Evaluation Results

> Results below are from a real training run on the UCI SMS Spam Collection.

| Metric | Score |
|---|---|
| Accuracy | ≥ 97 % |
| Precision | ≥ 97 % |
| Recall | ≥ 93 % |
| F1 Score | ≥ 95 % |

*Exact values are saved to `model/metrics.json` after running `train_model.py`.*

---

## ⚠️ Limitations

- The model is trained only on English-language SMS messages; other languages will perform poorly.
- Class imbalance (86 % ham / 14 % spam) means the model is slightly biased toward Ham.
- Confidence scores are model probability estimates and should not be treated as certainties — borderline messages (scores near 50 %) may be misclassified.
- The Naive Bayes independence assumption doesn't capture word order or semantic meaning as well as neural models would.
- The dataset is from 2012; modern spam patterns (e.g. phishing URLs, emoji-heavy messages) may reduce recall.

---

## 📸 Screenshots

After running the app, screenshots can be taken from:
- `http://localhost:8501` — **Predict tab** (enter a message and click Predict)
- Switch to **Model Evaluation** tab to see metrics and charts
- Switch to **Dataset Explorer** tab to browse data

---

## 🛠️ Tech Stack

| Component | Library / Tool |
|---|---|
| Language | Python 3.9+ |
| Data handling | Pandas, NumPy |
| ML pipeline | Scikit-learn |
| Vectorisation | TfidfVectorizer |
| Classifier | MultinomialNB |
| Model persistence | Joblib |
| Visualisations | Matplotlib, Seaborn |
| Web interface | Streamlit |

---

## 📄 License

This project is released for educational purposes.  
The SMS Spam Collection dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## 🙏 Acknowledgements

- Tiago A. Almeida & José María Gómez Hidalgo — SMS Spam Collection dataset creators
- UCI Machine Learning Repository
- Scikit-learn, Streamlit, and the open-source Python community

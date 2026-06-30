# Fake News Detection using Passive Aggressive Classifier

A machine learning project that classifies news articles as real or fake using TF-IDF vectorization and a Passive Aggressive Classifier, achieving ~99.5% accuracy.

## Getting Started

Open the notebook in Google Colab or Jupyter and run the cells in order.

## Use your preferred IDE

If you want to work locally using your own IDE, you can clone this repo and run the notebook.

The only requirement is having Python 3 installed.

Follow these steps:

```sh
# Step 1: Clone the repository.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
pip install pandas scikit-learn jupyter

# Step 4: Launch the notebook.
jupyter notebook Fake_News_Detection_using_PassiveAggressiveClassifier_main.ipynb
```

## Dataset

This project expects a `Fake.csv` (or similarly structured) dataset of news articles. When run in Colab, the notebook prompts a file upload at runtime via `google.colab.files`.

## What technologies are used for this project?

This project is built with:

- Python
- pandas
- scikit-learn (TfidfVectorizer, PassiveAggressiveClassifier)
- Jupyter / Google Colab

## Model performance

- Accuracy: ~99.5%
- Evaluated using a confusion matrix on the held-out test split

## How it works

1. Load and split the dataset into train/test sets
2. Convert article text into TF-IDF feature vectors
3. Train a `PassiveAggressiveClassifier` on the vectors
4. Evaluate predictions using accuracy score and confusion matrix

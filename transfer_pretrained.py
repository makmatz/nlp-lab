from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report

LABELS_MAPPING = {
    # --- MR models (binary: positive / negative) ---
    'siebert/sentiment-roberta-large-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'distilbert-base-uncased-finetuned-sst-2-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'textattack/bert-base-uncased-SST-2': {
        'LABEL_0': 'negative',
        'LABEL_1': 'positive',
    },
    'lvwerra/distilbert-imdb': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'aychang/roberta-base-imdb': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    # --- Semeval2017A models (3-class: positive / neutral / negative) ---
    'cardiffnlp/twitter-roberta-base-sentiment': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    },
    'finiteautomata/bertweet-base-sentiment-analysis': {
        'POS': 'positive',
        'NEU': 'neutral',
        'NEG': 'negative',
    },
    'cardiffnlp/twitter-roberta-base-sentiment-latest': {
        'negative': 'negative',
        'neutral': 'neutral',
        'positive': 'positive',
    },
    'cardiffnlp/twitter-xlm-roberta-base-sentiment': {
        'negative': 'negative',
        'neutral': 'neutral',
        'positive': 'positive',
    },
    'Seethal/sentiment_analysis_generic_dataset': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    },
}

DATASET_MODELS = {
    'MR': [
        'siebert/sentiment-roberta-large-english',
        'distilbert-base-uncased-finetuned-sst-2-english',
        'textattack/bert-base-uncased-SST-2',
        'lvwerra/distilbert-imdb',
        'aychang/roberta-base-imdb',
    ],
    'Semeval2017A': [
        'cardiffnlp/twitter-roberta-base-sentiment',
        'finiteautomata/bertweet-base-sentiment-analysis',
        'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'cardiffnlp/twitter-xlm-roberta-base-sentiment',
        'Seethal/sentiment_analysis_generic_dataset',
    ],
}


if __name__ == '__main__':
    for DATASET, models in DATASET_MODELS.items():
        if DATASET == "Semeval2017A":
            X_train, y_train, X_test, y_test = load_Semeval2017A()
        elif DATASET == "MR":
            X_train, y_train, X_test, y_test = load_MR()
        else:
            raise ValueError("Invalid dataset")

        le = LabelEncoder()
        le.fit(list(set(y_train)))
        y_test_enc = le.transform(y_test)

        for PRETRAINED_MODEL in models:
            print(f'\nRunning {PRETRAINED_MODEL} on {DATASET}...')
            sentiment_pipeline = pipeline("sentiment-analysis", model=PRETRAINED_MODEL)

            y_pred = []
            for x in tqdm(X_test):
                label = sentiment_pipeline(x)[0]['label']
                y_pred.append(LABELS_MAPPING[PRETRAINED_MODEL][label])

            y_pred_enc = le.transform(y_pred)
            print(f'\nDataset: {DATASET}\nPre-Trained model: {PRETRAINED_MODEL}\nTest set evaluation\n{get_metrics_report([y_test_enc], [y_pred_enc])}')
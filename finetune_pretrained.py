import numpy as np
import evaluate
from datasets import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from utils.load_datasets import load_MR, load_Semeval2017A


DATASETS = ['MR', 'Semeval2017A']
PRETRAINED_MODELS = ['bert-base-cased', 'roberta-base', 'distilbert-base-uncased']


metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def prepare_dataset(X, y):
    texts, labels = [], []
    for text, label in zip(X, y):
        texts.append(text)
        labels.append(label)

    return Dataset.from_dict({'text': texts, 'label': labels})


if __name__ == '__main__':

    results = {}

    for DATASET in DATASETS:
        # load the raw data
        if DATASET == "Semeval2017A":
            X_train, y_train, X_test, y_test = load_Semeval2017A()
        elif DATASET == "MR":
            X_train, y_train, X_test, y_test = load_MR()

        # encode labels
        le = LabelEncoder()
        le.fit(list(set(y_train)))
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
        n_classes = len(list(le.classes_))

        # prepare datasets
        train_set = prepare_dataset(X_train, y_train)
        test_set = prepare_dataset(X_test, y_test)

        for PRETRAINED_MODEL in PRETRAINED_MODELS:
            print(f"\n=== Dataset: {DATASET} | Model: {PRETRAINED_MODEL} ===")

            # define model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(
                PRETRAINED_MODEL, num_labels=n_classes)

            # tokenize datasets
            tokenized_train_set = train_set.map(tokenize_function)
            tokenized_test_set = test_set.map(tokenize_function)

            # training setup
            args = TrainingArguments(
                output_dir="output",
                eval_strategy="epoch",
                num_train_epochs=3,
                per_device_train_batch_size=32
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=tokenized_train_set,
                eval_dataset=tokenized_test_set,
                compute_metrics=compute_metrics,
            )

            # train and evaluate
            trainer.train()
            eval_results = trainer.evaluate()
            results[(DATASET, PRETRAINED_MODEL)] = eval_results
            print(f"Results: {eval_results}")

    # print summary table
    print("\n=== RESULTS SUMMARY ===")
    print(f"{'Dataset':<15} {'Model':<30} {'Accuracy':>10}")
    print("-" * 57)
    for (dataset, model_name), res in results.items():
        print(f"{dataset:<15} {model_name:<30} {res['eval_accuracy']:>10.4f}")

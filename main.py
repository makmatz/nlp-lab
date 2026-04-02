import argparse
import os
import warnings
import matplotlib
import sys

if '--no_show' in sys.argv:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from early_stopper import EarlyStopper
from models import BaselineDNN, LSTM
from attention import SimpleSelfAttentionModel, MultiHeadAttentionModel, TransformerEncoderModel
from training import train_dataset, eval_dataset, get_metrics_report, torch_train_val_split
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='transformer',
                    choices=['baseline_mean', 'baseline_mean_max', 'lstm', 'attention', 'multihead', 'transformer'],
                    help='Model to train')
parser.add_argument('--dataset', type=str, default='MR',
                    choices=['MR', 'Semeval2017A'],
                    help='Dataset to use')
parser.add_argument('--n_head', type=int, default=4,
                    help='Number of attention heads (transformer/multihead only)')
parser.add_argument('--n_layer', type=int, default=3,
                    help='Number of transformer encoder layers (transformer only)')
parser.add_argument('--no_show', action='store_true',
                    help='Save plots without displaying them')
args = parser.parse_args()
DATASET = args.dataset

# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
if DATASET == "MR":
    EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.200d.txt")
elif DATASET == "Semeval2017A":
    EMBEDDINGS = os.path.join(EMB_PATH, "glove.twitter.27B.200d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 200

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
original_labels = y_train[:10]

label_encoder = LabelEncoder()
label_encoder.fit(y_train)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
n_classes = label_encoder.classes_.size

encoded_labels = y_train[:10]

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx, dataset=DATASET)
test_set = SentenceDataset(X_test, y_test, word2idx, dataset=DATASET)

train_loader, val_loader = torch_train_val_split(train_set, BATCH_SIZE, BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

#############################################################################
# Model Definition, Training, and Evaluation
#############################################################################

output_size = 1 if n_classes == 2 else n_classes

models = {
    'baseline_mean':    BaselineDNN(output_size=output_size, embeddings=embeddings, trainable_emb=EMB_TRAINABLE, pooling='mean'),
    'baseline_mean_max':BaselineDNN(output_size=output_size, embeddings=embeddings, trainable_emb=EMB_TRAINABLE, pooling='mean_max'),
    'lstm':       LSTM(output_size=output_size, embeddings=embeddings, trainable_emb=EMB_TRAINABLE, bidirectional=True),
    'attention':  SimpleSelfAttentionModel(output_size=output_size, embeddings=embeddings, max_length=60),
    'multihead':  MultiHeadAttentionModel(output_size=output_size, embeddings=embeddings, max_length=60, n_head=args.n_head),
    'transformer':TransformerEncoderModel(output_size=output_size, embeddings=embeddings, max_length=60, n_head=args.n_head, n_layer=args.n_layer),
}

model = models[args.model]
print(f'Using model: {args.model}')

model.to(DEVICE)
print(model)

if n_classes == 2:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.CrossEntropyLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=1e-3)

early_stopper = EarlyStopper(model, save_path='best_model.pt', patience=5)

train_losses = []
val_losses = []
test_losses = []

for epoch in range(1, EPOCHS + 1):
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    train_loss, _ = eval_dataset(train_loader, model, criterion)
    val_loss, _ = eval_dataset(val_loader, model, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if early_stopper.early_stop(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break

# Load best model before evaluating
model.load_state_dict(torch.load('best_model.pt'))

_, (y_train_pred, y_train_gold) = eval_dataset(train_loader, model, criterion)
_, (y_test_pred, y_test_gold) = eval_dataset(test_loader, model, criterion)

print(f"\n{'='*50}")
print(f"Final Metrics on {DATASET}:")
print(f"\nTrain set:\n{get_metrics_report(y_train_gold, y_train_pred)}")
print(f"\nTest set:\n{get_metrics_report(y_test_gold, y_test_pred)}")

model_info = args.model
if args.model in ('transformer', 'multihead'):
    model_info += f' (n_head={args.n_head}'
    if args.model == 'transformer':
        model_info += f', n_layer={args.n_layer}'
    model_info += ')'

entry = (
    f'{"=" * 60}\n'
    f'Dataset:  {DATASET}\n'
    f'Model:    {model_info}\n'
    f'{"=" * 60}\n'
    f'Train set:\n{get_metrics_report(y_train_gold, y_train_pred)}\n'
    f'Test set:\n{get_metrics_report(y_test_gold, y_test_pred)}\n'
)
with open('report/results.txt', 'a') as f:
    f.write(entry)

# Plot loss curves
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training, Validation and Test Loss')
plt.legend()
plt.tight_layout()
os.makedirs('curves', exist_ok=True)
plt.savefig(f'curves/loss_curve_{args.model}_{DATASET}.png')
if not args.no_show:
    plt.show(block=False)

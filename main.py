import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from training import train_dataset, eval_dataset, get_metrics_report
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from utils.results import update_results

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.twitter.27B.200d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 200

EMB_TRAINABLE = False
BATCH_SIZE = 64
EPOCHS = 50
DATASET = "Semeval2017A"  # options: "MR", "Semeval2017A"

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

print('\nFirst 10 labels of the training set:')
for org, enc in zip(original_labels, encoded_labels):
    print(f"{org!r:12} -> {enc}")

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx, dataset=DATASET)

print('\nFirst 10 training examples:')
for i in range(10):
    print(f"[{i}] words={train_set.data[i]}, label={train_set.labels[i]}")
    print()

print('\nFirst 5 training examples (original vs encoded):')
for i in range(5):
    example, label, length = train_set[i]
    print(f"Original : {X_train[i]}")
    print(f"Encoded  : example={example}, label={label}, length={length}")
    print()

test_set = SentenceDataset(X_test, y_test, word2idx, dataset=DATASET)

# EX7 - Define our PyTorch-based DataLoader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
model = BaselineDNN(output_size=1 if n_classes==2 else n_classes,
                    hidden_size=64,  
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
if n_classes == 2:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.CrossEntropyLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=1e-3)

#############################################################################
# Training Pipeline
#############################################################################
train_losses = []
test_losses = []

best_test_loss = float('inf')
patience = 5
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)

    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Print final metrics
print(f"\n{'='*50}")
print(f"Final Metrics on {DATASET}:")
print(f"\nTrain set:\n{get_metrics_report(y_train_gold, y_train_pred)}")
print(f"\nTest set:\n{get_metrics_report(y_test_gold, y_test_pred)}")

# Save results
y_test_gold_flat = np.concatenate(y_test_gold)
y_test_pred_flat = np.concatenate(y_test_pred)
update_results(
    model_name='BaselineDNN',
    dataset=DATASET,
    hyperparams={
        'emb_dim': EMB_DIM,
        'emb_trainable': EMB_TRAINABLE,
        'batch_size': BATCH_SIZE,
        'epochs_trained': len(train_losses),
        'hidden_size': 256,
        'embeddings': os.path.basename(EMBEDDINGS),
    },
    metrics={
        'accuracy': accuracy_score(y_test_gold_flat, y_test_pred_flat),
        'f1': f1_score(y_test_gold_flat, y_test_pred_flat, average='macro'),
        'recall': recall_score(y_test_gold_flat, y_test_pred_flat, average='macro'),
    }
)

# Plot loss curves
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Test Loss - {DATASET}')
plt.legend()
plt.tight_layout()
plt.savefig(f'curves/loss_curve_{DATASET}.png')
plt.show()

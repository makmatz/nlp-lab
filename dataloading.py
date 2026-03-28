import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

text_processor = TextPreProcessor(
    normalize=['url', 'email', 'phone', 'user'],
    annotate={"hashtag", "elongated", "allcaps"},
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)

"""
Sentence length statistics (in tokens) on training set:
         max   mean  median  90th  => MAX_LENGTH
MR:       59   21.0    20.0    34  =>     35
Semeval:  67   25.4    26.0    34  =>     35
"""
MAX_LENGTH = {
    'MR': 35,
    'Semeval2017A': 35,
}

class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx, dataset='MR'):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
            dataset (str): the dataset name ('MR' or 'Semeval2017A')
        """

        self.max_length = MAX_LENGTH[dataset]
        self.labels = y
        self.word2idx = word2idx

        if dataset == 'MR':
            self.data = [sentence.lower().split() for sentence in X]
        elif dataset == 'Semeval2017A':
            self.data = [text_processor.pre_process_doc(sentence) for sentence in X]

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """

        fallback = self.word2idx['<unk>']
        indices = [self.word2idx.get(word, fallback) for word in self.data[index]]
        label = self.labels[index]
        length = len(self.data[index])

        if length > self.max_length:
            indices = indices[:self.max_length]
            length = self.max_length
        else:
            indices += [0] * (self.max_length - length)

        return torch.tensor(indices), torch.tensor(label), torch.tensor(length)
        


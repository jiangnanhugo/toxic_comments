import numpy as np
import re
import itertools
from collections import Counter
from collections import defaultdict, OrderedDict
import pandas as pd
import pickle as pkl

from nltk.stem import PorterStemmer
import h5py
import codecs

base_dir = './input/'
train_data_file = base_dir + 'train.csv'
test_data_file = base_dir + 'test.csv'

special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)
replace_numbers = re.compile(r'\d+', re.IGNORECASE)

stopwords = set(
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
     'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
     'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
     'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
     'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
     'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
     'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
     'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
     'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
     "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
     'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
     'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
     'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])


def sent_to_words(sent, remove_stopwords=False, stem_words=False):
    # print('-'*40)
    # print(sent)
    sent = sent.lower()
    sent = special_character_removal.sub('', sent)
    sent = replace_numbers.sub('num', sent)
    sent = sent.split()

    if remove_stopwords:
        sent = [w for w in sent if not w in stopwords]

    if stem_words:
        stemmer = PorterStemmer()
        sent = [stemmer.stem(w) for w in sent]

    return sent


def load_embedding(embedding_file):
    embedding_index = {}
    f = open(embedding_file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    print("Total {} words vector.".format(len(embedding_index)))
    return embedding_index


def read_csv(train_filepath=train_data_file, test_filepath=test_data_file, vocabulary_size=100000):
    train_df = pd.read_csv(train_filepath)
    test_df = pd.read_csv(test_filepath)
    texts = train_df["comment_text"].fillna("NA").values
    test_txt = test_df["comment_text"].fillna("NA").values
    listed_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    label = train_df[listed_classes].values

    comments = [sent_to_words(text) for text in texts]
    test_comments = [sent_to_words(text) for text in test_txt]

    vocab = defaultdict(float)
    for line in comments:
        for w in line:
            vocab[w] += 1

    print("len of all words {}".format(len(vocab)))
    word_to_count = sorted(vocab.items(), key=lambda t: t[1], reverse=True)
    print(word_to_count[0], word_to_count[vocabulary_size])
    truncat_vocab = word_to_count[:vocabulary_size]
    id2w = dict()
    w2id = dict()
    for ix, w in enumerate(truncat_vocab):
        id2w[ix] = w
        w2id[w] = ix

    sequences = [[w for w in line] for line in comments]
    test_sequences = [[w for w in line] for line in test_comments]

    fw = open('vocab.txt', 'w')
    trainf = codecs.open(base_dir + "train.txt", mode='w', encoding='utf-8')
    for line in sequences:
        trainf.write(" ".join(line) + '\n')
    trainf.flush()
    trainf.close()

    testf = codecs.open(base_dir + "test.txt", mode='w', encoding='utf-8')
    for line in test_sequences:
        testf.write(" ".join(line) + '\n')
    testf.flush()
    testf.close()


def clean_str(string):
    "tokenization/string for all dataset"
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", "")
    return string.strip().lower()


def load_data_and_labels(data_file):
    examples = list(open(data_file, 'r').readlines())
    examples = [s.strip() for s in examples]
    labels = [[]]
    return [examples, labels]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.range(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    read_csv()

import numpy as np
import re

from collections import defaultdict
import pandas as pd
import pickle


from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
import codecs

base_dir = './data/'
train_data_file = base_dir + 'train.csv'
test_data_file = base_dir + 'test.csv'

special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)
replace_numbers = re.compile(r'\d+', re.IGNORECASE)

SEPRATE_TOKEN = "#<TAB>#"

listed_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
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


def sent_to_words(sent, remove_stopwords=True, stem_words=False):
    sent = sent.strip().lower()
    sent = special_character_removal.sub('', sent)
    sent = replace_numbers.sub('num', sent)
    sent = sent.split()
    if remove_stopwords:
        sent = [w for w in sent if not w in stopwords]
    if stem_words:
        sent = [stemmer.stem(w) for w in sent]
    return sent


def crop_embedding(embedding_file, vocab_file):
    fw=codecs.open(embedding_file+'.crp','w',encoding='utf-8')
    embedding_index = {}
    word2index, _ = pickle.load(codecs.open(vocab_file, 'rb',encoding='utf-8'))
    f = codecs.open(embedding_file,encoding='utf-8')
    num=0
    printed=False
    visited=set()
    for line in f:
        values = line.strip().split()
        if len(values)<=2: break
        word = values[0]
        #coefs = np.asarray(values[1:], dtype='float32')
        if word in word2index:
            fw.write(word+"\t"+" ".join(values[1:])+'\n')
            num+=1
            visited.add(word)
  
    for w in word2index:
    	if w not in visited:
            random_vector = np.random.rand(300,1).flatten()
            if printed==False:
                print(" ".join([str(w) for w in random_vector]))
                printed=True
            fw.write(word+"\t"+" ".join([str(w) for w in random_vector])+'\n')
    fw.flush()
    fw.close()
    #     embedding_index[word] = coefs
    # lookup_table = []
    # num = 0
    # for w in word2index:
    #     if w in embedding_index:
    #         num += 1
    #         lookup_table.append(embedding_index[w])
    #     else:
    #         shape=embedding_index['unk'].shape
    #         random_vector=np.random.rand(*shape)
    #         print random_vector.shape
    #         lookup_table.append(random_vector)
    #
    # f.close()
    print("Total {}/{} words vector.".format(num, len(word2index)))

    #return lookup_table


def read_csv(train_filepath=train_data_file, test_filepath=test_data_file, vocabulary_size=100000):
    train_df = pd.read_csv(train_filepath)
    test_df = pd.read_csv(test_filepath)
    texts = train_df["comment_text"].fillna("NA").values
    label = train_df[listed_classes].values
    test_txt = test_df["comment_text"].fillna("NA").values

    comments = [sent_to_words(text) for text in texts]
    test_comments = [sent_to_words(text) for text in test_txt]

    vocab = defaultdict(float)
    for line in comments:
        for w in line: vocab[w] += 1

    print("len of all words {}".format(len(vocab)))
    word_to_count = sorted(vocab.items(), key=lambda t: t[1], reverse=True)
    print(word_to_count[0], word_to_count[vocabulary_size])
    truncat_vocab = word_to_count[:vocabulary_size]
    truncat_vocab.append(('unk', 1))
    id2w = dict()
    w2id = dict()
    for ix, w in enumerate(truncat_vocab):
        id2w[ix] = w[0]
        w2id[w[0]] = ix

    with open(base_dir + 'vocabulary.pkl', 'wb')as f:
        pickle.dump((w2id, id2w), f)

    trainf = codecs.open(base_dir + "train.txt", mode='w', encoding='utf-8')
    for line, lab in zip(comments, label):
        strnum = " ".join(str(num) for num in lab)
        trainf.write(" ".join(line) + SEPRATE_TOKEN + strnum + '\n')
    trainf.flush()
    trainf.close()

    testf = codecs.open(base_dir + "test.txt", mode='w', encoding='utf-8')
    for line in test_comments:
        testf.write(" ".join(line) + '\n')
    testf.flush()
    testf.close()


# def kflod(train_file):
#     data = open(train_file, 'r').read().split('\n')
#     texts = []
#     labels = []
#     for line in data:
#         splited = line.split(SEPRATE_TOKEN)
#         if len(splited) != 2:
#             print(splited)
#             break
#         texts.append(splited[0])
#         digits = splited[-1].split(' ')
#         labels.append([int(d) for d in digits])
#     print(len(texts), len(labels))
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
#     skf.get_n_splits(texts, labels)
#     print(skf)
#     idx = 0
#     for train_idx, valid_idx in skf.split(texts, labels):
#         fw = open(train_file + '.train.' + str(idx), 'w')
#         for idx in train_idx:
#             text, label = texts[idx], labels[idx]
#             fw.write(text + SEPRATE_TOKEN + " ".join(label) + '\n')
#         fw.close()
#         fw = open(train_file + ".valid." + str(idx), 'w')
#         for id in valid_idx:
#             text, label = texts[id], labels[id]
#             fw.write(text + SEPRATE_TOKEN + " ".join(label) + '\n')
#         fw.close()
#         idx += 1


def clean_str(string):
    "tokenization/string for all dataset"
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", "", string)
    return string.strip().lower()


def load_test_data(test_file, vocab_file, maxlen=150):
    data = open(test_file, 'r').readlines()
    print("len of data {}".format(len(data)))
    word2index, _ = pickle.load(open(vocab_file, 'r'))
    # print(word2index)
    texts = []
    print(len(data))
    for line in data:
        # print('-' * 40)
        # print(line)
        words = line.strip().split(' ')
        # print(len(words))
        # print('-' * 40)
        if len(words) == 0:
            break
        word_ids = np.zeros(maxlen, np.int32)
        for idx, w in enumerate(words):
            if idx >= maxlen: break
            if w in word2index:
                # print(w,word2index[w])
                word_ids[idx] = word2index[w]
            else:
                word_ids[idx] = word2index['unk']
        texts.append(word_ids)
    print(len(texts))
    return np.asarray(texts)


def load_data_and_labels(train_file, vocab_file, maxlen=150):
    data = open(train_file, 'r').read().split('\n')
    word2index, _ = pickle.load(open(vocab_file, 'r'))
    # print(word2index)
    texts = []
    labels = []
    for line in data:
        splited = line.split(SEPRATE_TOKEN)
        if len(splited) != 2:
            break
        word_ids = np.zeros(maxlen, np.int32)
        content = splited[0].split(' ')
        for idx, w in enumerate(content):
            if idx >= maxlen: break
            if w in word2index:
                word_ids[idx] = word2index[w]
            else:
                word_ids[idx] = word2index['unk']
        texts.append(word_ids)

        digits = splited[-1].split(' ')
        labels.append([int(d) for d in digits])

    return np.asarray(texts), np.asarray(labels)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    # kflod(base_dir+'train.txt')
    read_csv(vocabulary_size=100000)
    crop_embedding(base_dir + 'glove.840B.300d.txt', base_dir + "vocabulary.pkl")

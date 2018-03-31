# *_*coding:utf-8 *_* 
# Author:Aleck_
# @Time: 18-3-22 下午8:37
import torch
import torch.autograd as autograd
import codecs
import random
import torch.utils.data as Data

SEED = 1


# input: a sequence of tokens, and a token_to_index dictionary
# output: a LongTensor variable to encode the sequence of idxs
def prepare_sequence(seq, to_ix, cuda=False):
    # torch.LongTensor() 是一种包含单一数据类型元素的多维矩阵
    # 一个张量tensor可以从Python的list或序列构建
    # 此处，将这个句子中的所有词，在词典中的编号组成的list作为参数，最终返回这句话词的编号列表的张量
    var = autograd.Variable(torch.LongTensor([to_ix[w] for w in seq.split(' ')]))
    return var


def prepare_label(label, label_to_ix, cuda=False):
    var = autograd.Variable(torch.LongTensor([label_to_ix[label]]))
    return var


def build_token_to_ix(sentences):
    # 对应数据集的词库：存储数据集中的所有出现的不重复的词，且编号。
    token_to_ix = dict()
    print(len(sentences))
    for sent in sentences:
        for token in sent.split(' '):
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    token_to_ix['<pad>'] = len(token_to_ix)
    return token_to_ix


def build_label_to_ix(labels):
    label_to_ix = dict()
    for label in labels:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)


def load_MR_data():
    # already tokenized and there is no standard split
    # the size follow the Mou et al. 2016 instead
    file_pos = './datasets/MR/rt-polarity.pos'
    file_neg = './datasets/MR/rt-polarity.neg'
    print('loading MR datasets from', file_pos, 'and', file_neg)

    # codecs.open() :读入数据时，直接解码操作。防止编码格式问题。
    # .read()，.randlines(),.readline()这些方法区别是：
    # rand()读取整个文件成一个字符串
    # randlines()读取整个文件，自动将文件分成行的列表，共for line in fh.readlines():调用
    # randline() 读取一行数据，速度慢。通常在内存不够的情况下使用
    # split()分割字符串，返回分割后的字符串列表
    pos_sents = codecs.open(file_pos, 'r', 'utf8').read().split('\n')
    neg_sents = codecs.open(file_neg, 'r', 'utf8').read().split('\n')

    # seed()设置随机数种子，如果不了解原理可以不设置，python会自动设置好
    random.seed(SEED)
    random.shuffle(pos_sents)  # 随机洗牌，打乱顺序
    random.shuffle(neg_sents)

    print(len(pos_sents))
    print(len(neg_sents))

    # 将近80%的数据作为训练集，正向和负向各选取80%的数据集作为训练集
    train_data = [(sent, 1) for sent in pos_sents[:4250]] + [(sent, 0) for sent in neg_sents[:4250]]
    # 约10%的数据作为验证集
    dev_data = [(sent, 1) for sent in pos_sents[4250:4800]] + [(sent, 0) for sent in neg_sents[4250:4800]]
    # 约10%的数据作为测试集
    test_data = [(sent, 1) for sent in pos_sents[4800:]] + [(sent, 0) for sent in neg_sents[4800:]]

    # 随机洗牌，打乱顺序
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)

    print('train:', len(train_data), 'dev:', len(dev_data), 'test:', len(test_data))

    # [s ...]为所有评论句子组成的list
    word_to_ix = build_token_to_ix([s for s, _ in train_data + dev_data + test_data])
    label_to_ix = {0: 0, 1: 1}
    print('vocab size:', len(word_to_ix), 'label size:', len(label_to_ix))
    print('loading datasets done!')
    return train_data, dev_data, test_data, word_to_ix, label_to_ix


def load_MR_data_batch():
    pass


# train_data, dev_data, test_data, word_to_ix, label_to_ix = load_MR_data()
#
# var = prepare_sequence(train_data[0][0],word_to_ix)
# print(var)
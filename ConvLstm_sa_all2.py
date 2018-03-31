# *_*coding:utf-8 *_* 
# Author:Aleck_
# @Time: 18-3-23 上午9:24

# 实现2017 IEEE论文
# Deep Learning Approach for Sentiment Analysis of Short Texts

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import random
import torch.optim as optim
import os
import data_loader
import time

torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

SENTENCE_LENGTH = 61  # 为对应预料的最大长度。提前计算好的。
EMBEDDING_DIM = 50
HIDDEN_DIM = 50
EPOCH = 10
LR = 1e-3

time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
filename = "logs/out" + time + ".txt"
f = open(filename, "w")

print("====================== parameter ==================================\n")
print('ConvLstm_sa_all2.py')
print('filename: ' + filename)
print('''
    SENTENCE_LENGTH = 61  
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 50
    EPOCH = 100
    LR = 1e-3
''')

print("========================================================\n")

# 写入文件
print("====================== parameter ==================================\n", file=f)
print('ConvLstm_sa_all2.py', file=f)
print('filename: ' + filename, file=f)
print('''
    SENTENCE_LENGTH = 61  
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 50
    EPOCH = 100
    LR = 1e-3
''', file=f)

print("========================================================\n", file=f)


class ConvLstmSA(nn.Module):
    """
    一个卷积层+LSTM
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(ConvLstmSA, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.conv1 = nn.Conv2d(1, embedding_dim, (2, embedding_dim), padding=(0, 0), stride=(1, 1))

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        # 此处接受的句子编号序列，长度都已经一致了。通过词嵌入成相同维度的矩阵
        embeds = self.word_embeddings(sentence)

        in_embed = embeds.view(1, 1, len(sentence), -1)
        ls_inputs = self.conv1(in_embed)  # ls_inputs的维度为：embedding_dim*1*len(sentence)
        # 维度转化
        ls_inputs2 = torch.transpose(ls_inputs, 1, 3)

        ls_inputs3 = torch.transpose(ls_inputs2, 1, 2)

        lstm_out, lstm_hidden = self.lstm(ls_inputs3[0], self.hidden)

        # print("lstm_out:=======================================")
        # print(lstm_out)
        # print("lstm_hidden:===================================")
        # print(lstm_hidden)

        tag_space = self.hidden2tag(lstm_out[-1])
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def prepare_sequence(seq, to_ix):
    """
    返回句子长度为SENTENCE_LENGTH，不够的后面补0
    :param seq:
    :param to_ix:
    :return:
    """
    # idxs=[]
    # for w in seq:
    #     idxs.append(to_ix[w])
    #
    idxs = [to_ix[w] for w in seq.split(' ')]

    for _ in range(SENTENCE_LENGTH - len(idxs)):
        # idxs.append(0)
        idxs.insert(0,0)

    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)


def train():
    train_data, dev_data, test_data, word_to_ix, label_to_ix = data_loader.load_MR_data()

    train_data = train_data[:10]
    dev_data = dev_data[:5]
    test_data = test_data[:5]
    best_dev_acc = 0.0

    model = ConvLstmSA(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(label_to_ix))

    loss_function = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    no_up = 0
    for i in range(EPOCH):
        """
        每训练一个轮回，进行一次验证和测试
        """
        random.shuffle(train_data)
        print('epoch: %d start!' % i)
        print('epoch: %d start!' % i, file=f)

        train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i)
        print('now best dev acc:', best_dev_acc)
        print('now best dev acc:', best_dev_acc, file=f)

        dev_acc = evaluate(model, dev_data, loss_function, word_to_ix, label_to_ix, 'dev')
        test_acc = evaluate(model, test_data, loss_function, word_to_ix, label_to_ix, 'test')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            os.system('rm mr_best_model_acc_*.model')
            print('New Best Dev!!!')
            print('New Best Dev!!!', file=f)
            torch.save(model.state_dict(), 'best_models/mr_best_model_acc_' + str(int(test_acc * 10000)) + '.model')
            no_up = 0
        else:
            no_up += 1
            if no_up >= 10:
                exit()


def train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i, lr=1e-1):
    model.train()

    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    batch_sent = []

    for sent, label in train_data:
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        # sent = data_loader.prepare_sequence(sent, word_to_ix)
        label = data_loader.prepare_label(label, label_to_ix)

        sentence_in = prepare_sequence(sent, word_to_ix)

        tag_scores = model(sentence_in)

        pred_label = tag_scores.data.max(1)[1].numpy()
        pred_res.append(pred_label)

        # zero grad parameters
        model.zero_grad()

        loss = loss_function(tag_scores, label[0])
        avg_loss += loss.data[0]
        count += 1
        if count % 500 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count, loss.data[0]))
            print('epoch: %d iterations: %d loss :%g' % (i, count, loss.data[0]), file=f)



        # compute new grad parameters through time!
        loss.backward()

        # learning_rate step against the gradient
        # 根据学习率更新模型参数
        # for p in model.parameters():
        #     p.data.sub_(p.grad.data * lr)

        # 方案2 优化方法
        optimizer.step()

    avg_loss /= len(train_data)
    print('epoch: %d done! \n train avg_loss:%g , acc:%g' % (i, avg_loss, get_accuracy(truth_res, pred_res)))
    print('epoch: %d done! \n train avg_loss:%g , acc:%g' % (i, avg_loss, get_accuracy(truth_res, pred_res)), file=f)



def evaluate(model, data, loss_function, word_to_ix, label_to_ix, name='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0

    for sent, label in data:
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        # sent = data_loader.prepare_sequence(sent, word_to_ix)

        label = data_loader.prepare_label(label, label_to_ix)

        sentence_in = prepare_sequence(sent, word_to_ix)

        pred = model(sentence_in)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)

        # model.zero_grad() # should I keep this when I am evaluating the model?
        loss = loss_function(pred, label[0])
        avg_loss += loss.data[0]
        count += 1
        if count % 100 == 0:
            print('iterations: %d  evaluate loss :%g' % (count, loss.data[0]))
            print('iterations: %d  evaluate loss :%g' % (count, loss.data[0]), file=f)

    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc))
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc), file=f)
    return acc


train()

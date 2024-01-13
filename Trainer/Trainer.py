import paddle
import numpy as np
# from tqdm import trange
# from pgl.utils.logger import log
import paddle.nn.functional as F
import paddle.nn as nn
from sklearn.svm import SVC
from paddle.optimizer import Adam


class Classifier(nn.Layer):
    def __init__(self, hidden_size, num_class):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax()

    def forward(self, features):
        features = self.fc(features)
        return self.softmax(features)


class LogReg(nn.Layer):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            paddle.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def train_classifier_epoch(node_index, node_label, classifier, embeds, optim):
    classifier.train()
    pred = classifier(embeds)
    pred = paddle.gather(pred, node_index)
    loss = F.nll_loss(pred, node_label)
    loss.backward()
    acc = paddle.metric.accuracy(input=pred, label=node_label, k=1)
    optim.step()
    optim.clear_grad()
    return loss, acc


@paddle.no_grad()
def eval_node(node_index, node_label, classifier, embeds):
    classifier.eval()
    pred = classifier(embeds)
    pred = paddle.gather(pred, node_index)
    loss = F.nll_loss(pred, node_label)
    acc = paddle.metric.accuracy(input=pred, label=node_label, k=1)
    return loss, acc


class Trainer(object):
    def __init__(self, dataset, full_dataset, train_mask=None, val_mask=None, test_mask=None, classifier='LogReg',
                 metric='acc',
                 log_interval=1, **kwargs):
        self.data = dataset
        self.dataset = full_dataset
        self.classifier = classifier
        self.metric = metric
        self.log_interval = log_interval
        # self.train_mask = full_dataset[0].train_mask if train_mask is None else train_mask
        # self.val_mask = full_dataset[0].val_mask if val_mask is None else val_mask
        # self.test_mask = full_dataset[0].test_mask if test_mask is None else test_mask
        # self.setup_train_config(**kwargs)
        self.p_lr = 0.01
        self.weight_decay = 0.0

    def setup_train_config(self, p_optim='Adam', p_lr=0.01, runs=10, p_epoch=200, weight_decay=0.0, batch_szie=256):
        self.p_optim = p_optim
        self.runs = runs
        self.p_lr = p_lr
        self.p_epoch = p_epoch
        self.weight_decay = weight_decay
        self.batch_size = batch_szie

    def train_encoder(self, model):
        if model.mode == 'node_graph':
            train_fn = self.train_encoder_node_graph
        elif model.mode == 'graph':
            train_fn = self.train_encoder_graph
        elif model.mode == 'node':
            train_fn = self.train_encoder_node
        train_fn(model)

    def train_encoder_node(self, model):
        # if isinstance(model.encoders, list):
        #     params = [{'params': enc.parameters()}] for enc in model.encoders
        # else:
        #     params = model.encoders.parameters()
        p_optimizer = Adam(
            learning_rate=self.p_lr,
            parameters=model.parameters(),
            weight_decay=self.weight_decay)
        # p_optimizer = Adam(
        #     learning_rate=1e-3,
        #     parameters=model.parameters(),
        #     weight_decay=0.0)
        cal_train_loss = []
        best = 1e9
        wait_num = 0
        for epoch in range(self.p_epoch):
            loss = model.train_encoder_one_epoch(self.dataset, p_optimizer)
            loss = loss.item()
            print("epoch {:5d} | loss {:.4f}".format(epoch, loss))
            cal_train_loss.append(loss)
            if loss < best:
                wait_num = 0
                best = loss
            else:
                wait_num+=1
            # if wait_num>40:
            #     break
        print(" Model: DGI lowest train loss: %f" % (np.min(cal_train_loss)))
        self.train_classifier(self.data, model)

    def train_encoder_graph(self, model):
        # if isinstance(model.encoder, list):
        #     params = [{'params': enc.parameters()} for enc in model.encoder]
        # else:
        #     params = model.encoder.parameters()

        p_optimizer = Adam(
            learning_rate=self.p_lr,
            parameters=model.parameters(),
            weight_decay=self.weight_decay)
        best = 1e9
        wait_num = 0
        for run in range(self.runs):
            cal_train_loss = []
            for epoch in range(self.p_epoch):
                loss = model.train_encoder_one_epoch(self.dataset, p_optimizer)
                loss = loss.item()
                print("epoch {:5d} | loss {:.4f}".format(epoch, loss))
                cal_train_loss.append(loss)
                if loss < best:
                    wait_num = 0
                    best = loss
                else:
                    wait_num += 1
                if wait_num > 40:
                    break
        print(" Model: DGI lowest train loss: %f" % (np.min(cal_train_loss)))

    def train_encoder_node_graph(self, model):
        if isinstance(model.encoders, list):
            params = [{'params': enc.parameters()} for enc in model.encoders]
        else:
            params = model.encoders.parameters()

        p_optimizer = Adam(
            learning_rate=self.p_lr,
            parameters=params,
            weight_decay=self.weight_decay)
        best = 1e9
        wait_num = 0

        cal_train_loss = []
        for epoch in range(5):
            loss = model.train_encoder_one_epoch(self.dataset, p_optimizer)
            loss = loss.item()
            print("epoch {:05d} | loss {:.4f}".format(epoch, loss))
            cal_train_loss.append(loss)
            if loss < best:
                wait_num = 0
                best = loss
            else:
                wait_num += 1
            if wait_num > 20:
                break
        print(" Model: DGI lowest train loss: %f" % (np.min(cal_train_loss)))
        self.train_classifier(self.data, model)

    def train_classifier(self, dataset, model):
        graph = dataset.graph
        train_index = dataset.train_index
        train_label = dataset.train_label
        val_index = dataset.val_index
        val_label = dataset.val_label
        test_index = dataset.test_index
        test_label = dataset.test_label
        embeds = model.encoders[1](graph, graph.node_feat[list(graph.node_feat.keys())[0]])
        embeds = embeds.detach()
        # classifier = self.get_clf(embeds.shape[1])
        classifier = Classifier(embeds.shape[1], dataset.num_classes)
        optim = Adam(
            learning_rate=0.01,
            parameters=classifier.parameters(),
            weight_decay=self.weight_decay)
        cal_val_acc = []
        cal_test_acc = []
        cal_val_loss = []
        cal_test_loss = []
        for epoch in range(1200):
            train_classifier_epoch(train_index, train_label, classifier, embeds, optim)
            val_loss, val_acc = eval_node(val_index, val_label, classifier, embeds)
            cal_val_acc.append(val_acc.item())
            cal_val_loss.append(val_loss.item())
            print("Epoch {:05d} | val_loss {:.4f} | val_acc {:.4f}".format(epoch, val_loss.item(), val_acc.item()))

        test_loss, test_acc = eval_node(test_index, test_label, classifier, embeds)
        cal_test_acc.append(test_acc.item())
        cal_test_loss.append(test_loss.item())
        print("Model: DGI Best Test Accuracy: %f" % max(cal_test_acc))
        print("Model: DGI Average Test Accuracy: %f" %np.mean(cal_test_acc))


def get_clf(self, hid_units):
    if self.classifier == 'SVC':
        return SVC(C=10)
    elif self.classifier == 'LogReg':
        return LogReg(hid_units, self.num_classes)
    else:
        return None

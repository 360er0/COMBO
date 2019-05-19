import re
import os
import random

import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K


def ensure_deterministic():
    seed = 123
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    random.seed(seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)

    tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


class Token:

    def __init__(self, fields):
        self.fields = fields

    def __eq__(self, other_token):
        return self.fields == other_token.fields

    def __str__(self):
        return self.fields['form']

    def __repr__(self):
        return self.__str__()


class Tree:

    def __init__(self, tree_id, tokens, words, comments=None, probs=None, emb=None):
        self.id = tree_id
        self.tokens = tokens
        self.words = words
        self.comments = comments
        self.probs = probs
        self.emb = emb

    def __eq__(self, other_tree):
        return all([t1 == t2 for t1, t2 in zip(self.tokens, other_tree.tokens)])

    def __str__(self):
        return ' '.join(map(str, self.tokens))

    def __repr__(self):
        return self.__str__()


class TSVLoader:

    def safe_int(self, i):
        try:
            return int(i)
        except ValueError:
            return 0

    def load(self, filename):
        tree_id = 0
        trees = []
        tree = None
        comments = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                ls = line.strip().split('\t')
                
                if line == '\n':
                    if max([self.safe_int(t.fields['head']) for t in tree.tokens]) < len(tree.tokens):
                        trees.append(tree)
                    tree = None
                    continue
                
                if line[0] == '#':
                    comments.append(line.strip('\n'))
                    continue
                
                if tree is None:
                    tree = Tree(
                        tree_id=tree_id,
                        tokens=[],
                        words=[],
                        comments=comments,
                    )
                    tree_id += 1
                    comments = []

                    fields = dict(zip(self.columns, ['__ROOT__']*len(self.columns)))
                    fields['id'] = '0' 
                    fields['head'] = '0'

                    token = Token(
                        fields=fields,
                    )
                    tree.tokens.append(token)

                token = Token(
                    fields=dict(zip(self.columns, ls)),
                )
                
                if '-' in ls[0] or '.' in ls[0]:
                    tree.words.append(token)
                else:
                    tree.tokens.append(token)

            if tree is not None:
                if max([self.safe_int(t.fields['head']) for t in tree.tokens]) < len(tree.tokens):
                    trees.append(tree)

        return trees


class ConllLoader(TSVLoader):
    
    columns = [
        'id',
        'form',
        'lemma',
        'upostag',
        'xpostag',
        'feats',
        'head',
        'deprel',
        'deps',
        'misc',
    ]


class ConllSemanticLoader(TSVLoader):
    
    columns = [
        'id',
        'form',
        'lemma',
        'upostag',
        'xpostag',
        'feats',
        'head',
        'deprel',
        'deps',
        'misc',
        'semrel',
    ]


class TxtLoader():

    columns = [
        'id',
        'form',
        'lemma',
        'upostag',
        'xpostag',
        'feats',
        'head',
        'deprel',
        'deps',
        'misc',
    ]

    @staticmethod
    def tokenize(s):
        return [t for t in re.findall(r'\w+|\W', s) if ' ' not in t]

    def load(self, filename):
        output = []
        with open(filename, 'r', encoding='utf-8') as f:
            for tree_id, sent in enumerate(f):
                tree = Tree(
                    tree_id=tree_id,
                    tokens=[],
                    words=[],
                )
                
                fields = dict(zip(columns, ['__ROOT__']*len(columns)))
                fields['id'] = '0' 
                fields['head'] = '0'

                token = Token(
                    fields=fields,
                )
                tree.tokens.append(token)
                
                for token in self.tokenize(sent):
                    fields = dict(zip(columns, ['_']*len(columns)))
                    fields['form'] = token
                    token = Token(
                        fields=fields,
                    )
                    tree.tokens.append(token)
                    
                output.append(tree)
            
        return output


class TSVSaver:

    def save(self, filename, trees):
        with open(filename, 'w', encoding='utf-8') as f:
            for tree in trees:
                tree_output = []
                tree_output += tree.comments
                for token in sorted(
                    tree.words + tree.tokens[1:], 
                    key=lambda x: float(x.fields['id'].split('-')[0]),
                ):
                    line_output = []
                    for col in self.columns:
                        line_output.append(token.fields.get(col, '_'))
                    tree_output.append('\t'.join(line_output))
                f.write('\n'.join(tree_output) + '\n\n')


class ConllSaver(TSVSaver):

    columns = [
        'id',
        'form',
        'lemma',
        'upostag',
        'xpostag',
        'feats',
        'head',
        'deprel',
        'deps',
        'misc',
    ]


class ConllSemanticSaver(TSVSaver):

    columns = [
        'id',
        'form',
        'lemma',
        'upostag',
        'xpostag',
        'feats',
        'head',
        'deprel',
        'deps',
        'misc',
        'semrel',
    ]


class EmbeddingSaver:

    def save(self, filename, trees):
        with open(filename, 'w', encoding='utf-8') as f:
            for tree in trees:
                f.write(' '.join([tree.id] + map(str, tree.emb.tolist())) + '\n')


def accuracy_score(pred, true, fields):
    if len(pred) != len(true):
        raise ValueError

    correct = 0
    total = 0
    for p_tree, t_tree in zip(pred, true):
        for pred_token, true_token in zip(p_tree.tokens[1:], t_tree.tokens[1:]):
            same = True
            for field in fields:
                same = same and pred_token.fields.get(field) == true_token.fields.get(field)

            if same:
                correct += 1
            total += 1

    return correct/total


def feat_score(pred, true):
    if len(pred) != len(true):
        raise ValueError

    correct = 0
    total = 0
    for p_tree, t_tree in zip(pred, true):
        for pred_token, true_token in zip(p_tree.tokens[1:], t_tree.tokens[1:]):
            pred_feats = set(pred_token.fields['feats'].split('|'))
            true_feats = set(true_token.fields['feats'].split('|'))

            if pred_feats == true_feats:
                correct += 1
            total += 1

    return correct/total


def em_score(pred, true):
    if len(pred) != len(true):
        raise ValueError

    correct = 0
    total = 0
    for p, t in zip(pred, true):
        if p == t:
            correct += 1
        total += 1

    return correct/total


def cycle_score(pred, true):
    # https://sci-hub.tw/10.1002/aic.690110316

    d = max([len(t.tokens) for t in pred])

    pred = [[int(t.fields['head']) for t in tree.tokens] for tree in pred]
    pred = pad_sequences(pred, padding='post')
    pred = to_categorical(pred, num_classes=d)
    pred = pred[:, 1:, 1:]

    results = np.zeros(pred.shape[0])

    pred_n = pred
    for i in range(d - 1):
        results += np.sum(np.sum(pred_n*np.eye(d - 1), axis=1), axis=1)
        pred_n = pred_n @ pred

    return np.mean(results > 0.0)


def print_summary(pred, true):
    print('UAS: {}\nLAS: {}\nLEMMA: {}\nPOS: {}\nXPOS: {}\nFEAT: {}\nSEM: {}\nEM: {}\n'.format(
            accuracy_score(pred, true, ['head']),
            accuracy_score(pred, true, ['head', 'deprel']),
            accuracy_score(pred, true, ['lemma']),
            accuracy_score(pred, true, ['upostag']),
            accuracy_score(pred, true, ['xpostag']),
            feat_score(pred, true),
            accuracy_score(pred, true, ['semrel']),
            accuracy_score(pred, true, ['head', 'deprel', 'upostag', 'feat', 'lemma']),
        ))

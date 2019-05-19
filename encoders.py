import gzip
from collections import OrderedDict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from mst import mst


class EmbeddingLoader():

    def __init__(self, params):
        self.params = params
        self.word2idx = {
            '__PADDING__': 0,
            '__UNKNOWN__': 1,
            '__ROOT__': 2,
        }
        self.offset = len(self.word2idx)
        self.word_vectors = None
        self.embed_size = None
        self.vocab_size = None

    def load_embedding(self, embedding_file):
        if embedding_file.endswith('.gz'):
            f = gzip.open(embedding_file, 'rb')
        else:
            f = open(embedding_file, 'rb')

        n_row, n_col = map(int, f.readline().strip().split())

        self.embed_size = n_col
        self.vocab_size = n_row + self.offset
        self.word_vectors = np.zeros((self.vocab_size, self.embed_size))

        idx = self.offset
        for line in f:
            try:
                line = line.decode('utf-8')
            except UnicodeDecodeError:
                print('unicode error', line.split()[0])

            try:
                ls = line.strip().split()
                word = ls[0]
                if self.params.lower:
                    word = word.lower()

                vector = np.array(ls[1:])
                self.word2idx[word] = idx
                self.word_vectors[idx, :] = vector
            except:
                print('other error', word)

            idx += 1

        self.word_vectors[self.word2idx['__UNKNOWN__']] \
             = np.random.normal(
                np.mean(self.word_vectors),
                np.std(self.word_vectors),
                self.embed_size,
            )

        f.close()


class WordEmbedEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, params):
        self.params = params
        self.emb = self.load_emb()
        self.word_count = {}
        self.vocab_size = self.emb.vocab_size

    def load_emb(self):
        emb = EmbeddingLoader(params=self.params)
        emb.load_embedding(
            self.params.embed_file,
        )

        return emb

    def fit(self, trees, *args):
        for tree in trees:
            for token in tree.tokens:
                word = token.fields['form']
                if word not in self.word_count:
                    self.word_count[word] = 0
                self.word_count[word] += 1
        return self

    def transform(self, trees):
        out_rows = []
        for tree in trees:
            out_row = []
            for token in tree.tokens:
                word = token.fields['form']
                if self.params.lower:
                    word = word.lower()

                if word not in self.emb.word2idx:# or self.word_count.get(word, 0) < 2:
                    word = '__UNKNOWN__'
                out_row.append(self.emb.word2idx[word])
            out_rows.append(out_row)
        return out_rows


class OneHotEncoder(BaseEstimator, TransformerMixin):

    input_field = ''

    def __init__(self, params):
        self.params = params
        self.vocab_size = None
        self.pos2idx = {
            '__PADDING__': 0,
            '__UNKNOWN__': 1,
            '__ROOT__': 2,
        }
        self.idx2pos = {
            0: '__PADDING__',
            1: '__UNKNOWN__',
            2: '__ROOT__',
        }

    def fit(self, trees, *args):
        idx = len(self.pos2idx)
        for tree in trees:
            for token in tree.tokens:
                pos = token.fields[self.input_field]
                if pos not in self.pos2idx:
                    self.pos2idx[pos] = idx
                    self.idx2pos[idx] = pos
                    idx += 1
        self.vocab_size = len(self.pos2idx)

        return self

    def transform(self, trees):
        out_rows = []
        for tree in trees:
            out_row = []
            for token in tree.tokens:
                pos = token.fields[self.input_field]
                if pos not in self.pos2idx:
                    pos = '__UNKNOWN__'
                out_row.append(self.pos2idx[pos])
            out_rows.append(out_row)

        return out_rows

    def inverse_transform(self, pred, trees):
        pred = np.argmax(pred, axis=2)

        out_rows = []
        for pred_row in pred:
            out_row = []
            for idx in pred_row:
                out_row.append(self.idx2pos[idx])
            out_rows.append(out_row)

        return out_rows


class PosEncoder(OneHotEncoder):

    input_field = 'upostag'


class XposEncoder(OneHotEncoder):

    input_field = 'xpostag'


class DeprelEncoder(OneHotEncoder):

    input_field = 'deprel'


class WordEncoder(OneHotEncoder):

    input_field = 'form'


class SemrelEncoder(OneHotEncoder):

    input_field = 'semrel'


class FeatEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, params):
        self.params = params

        self.threshold = 0.5
        self.separator = '|'
        self.assigment = '='

        self.slices = {}
        self.feat2idx = {}
        self.idx2feat = {}

        self.vocab_size = None

    def fit(self, trees, *args):
        idx = len(self.feat2idx)

        # gather all (cat, val) pairs
        for tree in trees:
            for token in tree.tokens:
                for feat in token.fields['feats'].split(self.separator):
                    if self.assigment in feat:
                        cat, val = feat.split(self.assigment)
                    else:
                        cat, val = feat, feat
                    
                    if cat not in self.feat2idx:
                        self.feat2idx[cat] = {
                            None: 0,
                        }
                    self.feat2idx[cat][val] = 0

        # update ids
        for cat, vals in self.feat2idx.items():
            min_idx = idx
            for val in vals.keys():
                self.feat2idx[cat][val] = idx
                self.idx2feat[idx] = (cat, val)
                idx += 1

            self.slices[cat] = (min_idx, idx)

        self.vocab_size = idx + 1

        return self

    def transform(self, trees):
        out_rows = []
        for tree in trees:
            out_row = []
            for token in tree.tokens:
                out_feat = [0.0 for _ in range(self.vocab_size)]

                # load token feats
                token_feats_dict = {}
                for feat in token.fields['feats'].split(self.separator):
                    if self.assigment in feat:
                        cat, val = feat.split(self.assigment)
                    else:
                        cat, val = feat, feat

                    token_feats_dict[cat] = val

                # update vector
                for cat, val2idx in self.feat2idx.items():
                    val = token_feats_dict.get(cat, None)
                    out_feat[val2idx[val]] = 1.0

                out_row.append(out_feat)
            out_rows.append(out_row)
        return out_rows

    def inverse_transform(self, pred, trees):
        out_rows = []
        for sent in pred:
            out_row = []
            for word in sent:
                pred_feat = []
                for cat, (min_idx, max_idx) in self.slices.items():
                    best_idx = np.argmax(word[min_idx:max_idx]) + min_idx

                    cat, val = self.idx2feat[best_idx]
                    if val is not None:
                        if cat != val:
                            pred_feat.append(cat + self.assigment + val)
                        else:
                            pred_feat.append(val)

                if len(pred_feat) == 0:
                    pred_feat_str = '_'
                else:
                    pred_feat_str = self.separator.join(sorted(pred_feat))

                out_row.append(pred_feat_str)
            out_rows.append(out_row)
        return out_rows


class CharEncoder(BaseEstimator, TransformerMixin):

    input_field = ''

    def __init__(self, params):
        self.params = params
        self.vocab_size = None
        self.char2idx = {
            '__PADDING__': 0,
            '__UNKNOWN__': 1,
            '__ROOT__': 2,
            '__START__': 3,
            '__END__': 4,
        }
        self.idx2char = {
            0: '__PADDING__',
            1: '__UNKNOWN__',
            2: '__ROOT__',
            3: '__START__',
            4: '__END__',
        }

    def fit(self, trees, *args):
        idx = len(self.char2idx)
        for tree in trees:
            for token in tree.tokens:
                for char in token.fields[self.input_field]:
                    if char not in self.char2idx:
                        self.char2idx[char] = idx
                        self.idx2char[idx] = char
                        idx += 1
        self.vocab_size = len(self.char2idx)

        return self

    def transform(self, trees):
        out_rows = []
        for tree in trees:
            out_row = []
            for token in tree.tokens:
                word = token.fields[self.input_field]

                out_word = []
                out_word.append(self.char2idx['__START__'])
                for i in range(self.params.char_max_len + 1):
                    if word == '__ROOT__':
                        char = '__ROOT__'
                    elif i < len(word):
                        char = word[i]
                    elif i == len(word):
                        char = '__END__'
                    else:
                        char = '__PADDING__'

                    if char not in self.char2idx:
                        char = '__UNKNOWN__'

                    out_word.append(self.char2idx[char])
                out_row.append(out_word)
            out_rows.append(out_row)
        return out_rows

    def inverse_transform(self, pred, trees):
        pred = np.argmax(pred, axis=3)

        out_rows = []
        for pred_row in pred:
            out_row = ['__ROOT__']
            for pred_word in pred_row[1:]:
                out_word = []
                for idx in pred_word[1:-1]:
                    pred_char = self.idx2char[idx]

                    if pred_char == '__END__':
                        break
                    elif pred_char == '__PADDING__':
                        continue
                    elif '__' in pred_char:
                        pred_char = '?'
                    
                    out_word.append(pred_char)

                out_row.append(''.join(out_word))
            out_rows.append(out_row)

        return out_rows


class WordCharEncoder(CharEncoder):

    input_field = 'form'


class LemmaEncoder(CharEncoder):

    input_field = 'lemma'


class HeadEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, params):
        self.params = params

    def fit(self, trees, *args):
        return self

    def transform(self, trees):
        out_rows = []
        for tree in trees:
            out_row = []
            for token in tree.tokens:
                if token.fields['head'] == '_':
                    out_row.append(0)
                else:
                    out_row.append(int(token.fields['head']))
            out_rows.append(out_row)
        return out_rows

    def inverse_transform(self, pred, trees):
        if self.params.force_trees:
            output = []
            for i, tree in enumerate(trees):
                probs = pred[i, :, :].copy()
                n = len(trees[i].tokens)

                # make sure we won't predict padding as a head
                probs = probs[:n, :n]

                # choose the best tree
                heads = mst(probs)
                output.append(heads.astype(str))

            return output
        else:
            return np.argmax(pred, axis=2).astype(str)


class SentEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, params):
        self.params = params

    def fit(self, trees, *args):
        return self

    def transform(self, trees):
        return None

    def inverse_transform(self, pred, trees):
        return pred


class Factory:

    def __init__(self, params):
        self.params = params
        self.encoders = self.get_encoders()

    def get_encoders(self):
        raise NotImplementedError

    def fit(self, trees):
        for name, encoder in self.encoders.items():
            self.encoders[name] = encoder.fit(trees)
        return self

    def transform(self, trees):
        output = []
        for encoder in self.encoders.values():
            encoded_trees = encoder.transform(trees)
            if encoded_trees is not None:
                output.append(encoded_trees)

        return output

    def inverse_transform(self, preds, trees):
        output = []
        for pred, encoder in zip(preds, self.encoders.values()):
            output.append(encoder.inverse_transform(pred, trees))

        return output


class FeaturesFactory(Factory):

    def get_encoders(self):
        encoder_map = {
            'form': WordEmbedEncoder if self.params.embed_file is not None else WordEncoder,
            'lemma': LemmaEncoder,
            'upostag': PosEncoder,
            'xpostag': XposEncoder,
            'feats': FeatEncoder,
            'char': WordCharEncoder,
        }

        encoders = OrderedDict()
        for encoder_name in self.params.features:
            encoders[encoder_name] = encoder_map[encoder_name](self.params)

        return encoders


class TargetsFactory(Factory):

    def get_encoders(self):
        encoder_map = {
            'head': HeadEncoder,
            'deprel': DeprelEncoder,
            'lemma': LemmaEncoder,
            'upostag': PosEncoder,
            'xpostag': XposEncoder,
            'feats': FeatEncoder,
            'sent': SentEncoder,
            'semrel': SemrelEncoder,
        }

        encoders = OrderedDict()
        for encoder_name in self.params.targets:
            encoders[encoder_name] = encoder_map[encoder_name](self.params)

        return encoders

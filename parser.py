import random
from copy import deepcopy

import numpy as np
from sparse import COO
from sklearn.base import BaseEstimator, TransformerMixin
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from utils import Tree
from models import (
    KerasModel,
    ParserModel,
)
from encoders import (
    FeaturesFactory,
    TargetsFactory,
)


class Parser(BaseEstimator, TransformerMixin, KerasModel):

    def __init__(self, params):
        self.params = params
        self.features_factory = FeaturesFactory(self.params)
        self.targets_factory = TargetsFactory(self.params)
        self.model = None

    def create(self):
        return ParserModel(
            self.params,
            self.features_factory,
            self.targets_factory,
        ).model

    def create_generator(self, batches, multiple):
        if not multiple:
            for batch in batches:
                yield batch
        else:
            n_batches = len(batches)
            batch_idx = 0
            while True:
                yield batches[batch_idx]
                batch_idx = (batch_idx + 1) % n_batches

    def batchify_X(self, trees):
        raw = self.features_factory.transform(trees)
        output = []

        words_batch = 0
        n_cols = len(raw)

        batch = [[] for _ in range(n_cols)]
        for row_idx, tree in enumerate(trees):
            words_batch += len(tree.tokens)

            for col_idx in range(n_cols):
                batch[col_idx].append(raw[col_idx][row_idx])

            if words_batch > self.params.batch_size:
                output_batch = [pad_sequences(x, padding='post') for x in batch]
                batch = [[] for _ in range(n_cols)]
                output.append(output_batch)
                words_batch = 0

        if words_batch > 0:
                output_batch = [pad_sequences(x, padding='post') for x in batch]
                output.append(output_batch)

        return output

    def batchify_y(self, trees):
        raw = self.targets_factory.transform(trees)
        output = []

        words_batch = 0
        n_cols = len(raw)

        batch = [[] for _ in range(n_cols)]
        for row_idx, tree in enumerate(trees):
            words_batch += len(tree.tokens)

            for col_idx in range(n_cols):
                batch[col_idx].append(raw[col_idx][row_idx])

            if words_batch > self.params.batch_size:
                padded_batch = [pad_sequences(x, padding='post') for x in batch]
                output_batch = []

                for target, padded_target in zip(self.params.targets, padded_batch):
                    if target == 'head':
                        output_batch.append(to_categorical(padded_target, num_classes=padded_target.shape[1]))
                    elif target in ['feats', 'sent']:
                        output_batch.append(padded_target)
                    else:
                        output_batch.append(to_categorical(padded_target, num_classes=self.targets_factory.encoders[target].vocab_size))

                batch = [[] for _ in range(n_cols)]
                output.append([COO.from_numpy(a) for a in output_batch])
                words_batch = 0

        if words_batch > 0:
            padded_batch = [pad_sequences(x, padding='post') for x in batch]
            output_batch = []

            for target, padded_target in zip(self.params.targets, padded_batch):
                if target == 'head':
                    output_batch.append(to_categorical(padded_target, num_classes=padded_target.shape[1]))
                elif target in ['feats', 'sent']:
                    output_batch.append(padded_target)
                else:
                    output_batch.append(to_categorical(padded_target, num_classes=self.targets_factory.encoders[target].vocab_size))

            batch = [[] for _ in range(n_cols)]
            output.append([COO.from_numpy(a) for a in output_batch])

        return output

    def batchify_weights(self, trees):
        output = []
        words_batch = 0
        targets = [t for t in self.params.targets if t not in ['sent']]
        n_cols = len(targets)
        batch = [[] for _ in range(n_cols)]
        for row_idx, tree in enumerate(trees):
            words_batch += len(tree.tokens)
            sample_weight = np.log(len(tree.tokens))
            if not self.params.train_partial or self.params.full_tree in tree.comments:
                nonzero_targets = {
                    'head',
                    'deprel',
                    'lemma',
                    'upostag',
                    'xpostag',
                    'feats',
                    'semrel',
                }
            elif self.params.partial_tree in tree.comments:
                nonzero_targets = {
                    'lemma',
                    'upostag',
                    'xpostag',
                    'feats',
                }
            else:
                nonzero_targets = set()

            for col_idx, target in enumerate(targets):
                batch[col_idx].append(sample_weight if target in nonzero_targets else 1e-9)

            if words_batch > self.params.batch_size:
                output.append(batch)
                batch = [[] for _ in range(n_cols)]
                words_batch = 0

        if words_batch > 0:
            output.append(batch)
            batch = [[] for _ in range(n_cols)]

        return output

    def fit(self, trees, shuffle=True):
        trees = sorted(trees, key=lambda x: len(x.tokens))

        if self.model is None:
            self.features_factory = self.features_factory.fit(trees)
            self.targets_factory = self.targets_factory.fit(trees)
            self.model = self.create()

        batches = list(zip(
                self.batchify_X(trees), 
                self.batchify_y(trees),
                self.batchify_weights(trees),
            ),
        )

        try:
            for epoch_idx in range(self.params.epochs):
                if shuffle:
                    random.shuffle(batches)

                for batch_idx, batch in enumerate(batches):
                    losses = self.model.train_on_batch(
                        x=batch[0],
                        y=[a.todense() for a in batch[1]],
                        sample_weight=[np.array(w) for w in batch[2]],
                        # class_weight=['auto']*len(self.params.targets),
                    )
                    if not isinstance(losses, list):
                        losses = [losses]

                    print(epoch_idx, batch_idx, list(zip(self.model.metrics_names, losses)))

        except KeyboardInterrupt:
            pass

    def predict(self, trees):
        trees = sorted(trees, key=lambda x: len(x.tokens))
        output_trees = []

        tree_idx = 0
        for batch in self.batchify_X(trees):
            batch_trees = trees[tree_idx:(tree_idx + batch[0].shape[0])]
            batch_probs = self.model.predict_on_batch(batch)
            if not isinstance(batch_probs, list):
                batch_probs = [batch_probs]
            batch_preds = self.targets_factory.inverse_transform(batch_probs, batch_trees)

            for row_idx, old_tree in enumerate(batch_trees):
                row_probs = [p[row_idx] for p in batch_probs]
                row_preds = [p[row_idx] for p in batch_preds]

                emb = None
                new_tokens = []
                for token_idx, token in enumerate(old_tree.tokens):
                    new_token = deepcopy(token)
                    for field, pred in zip(self.params.targets, row_preds):
                        if field == 'sent':
                            emb = pred
                        else:
                            new_token.fields[field] = pred[token_idx]
                    new_tokens.append(new_token)

                output_trees.append(Tree(
                        tree_id=old_tree.id, 
                        tokens=new_tokens,
                        words=old_tree.words,
                        comments=old_tree.comments,
                        probs=row_probs if self.params.save_probs else None,
                        emb=emb,
                    ))
                tree_idx += 1

        output_trees = sorted(output_trees, key=lambda x: x.id)

        return output_trees

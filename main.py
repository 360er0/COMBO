### set random seed + limit number of threads

import os
import time
import random

import numpy as np
import tensorflow as tf


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(123)
random.seed(123)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)

from keras import backend as K
tf.set_random_seed(123)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


### main

from argparse import ArgumentParser
from sklearn.externals import joblib
from keras import backend as K

from parser import Parser
from utils import (
    ConllLoader,
    ConllSaver,
    ConllSemanticLoader,
    ConllSemanticSaver,
    print_summary,
    uas_score,
    las_score,
    lemma_score,
    pos_score,
    xpos_score,
    semrel_score,
    feat_score,
    full_score,
)


def valid_params(params):
    if 'deprel' in params.targets and 'head' not in params.targets:
        raise KeyError('You have to predict "head" in order to predict "deprel".')

    if params.mode not in ['train', 'autotrain', 'multitrain', 'evaluate', 'predict', 'multipredict']:
        raise KeyError('Set "mode" argument to either "train", "predict" or "evaluate".')

    if len(params.targets) != len(params.loss_weights):
        raise KeyError('loss_weights and targets must be the same length.')


def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))


def get_comma_separated_float_args(option, opt, value, parser):
    setattr(parser.values, option.dest, [float(v) for v in value.split(',')])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", dest="mode", help="Mode of parser (train or predict)",
        choices=['train', 'autotrain', 'multitrain', 'evaluate', 'predict',
                 'multipredict'])

    parser.add_argument("--train", dest="train",
                      help="Annotated CONLLu train file", metavar="FILE")
    parser.add_argument("--valid", dest="valid",
                      help="Annotated CONLLu valid file", metavar="FILE")
    parser.add_argument("--test", dest="test",
                      help="Unannotated CONLLu test file", metavar="FILE")

    parser.add_argument("--embed", dest="embed_file",
                      help="External embeddings for forms", metavar="FILE")
    parser.add_argument("--model", dest="model_file", default="model.pkl",
                      help="Load/Save model file", metavar="FILE")
    parser.add_argument("--pred", dest="pred_file",
                      help="CONLLu output pred file", metavar="FILE")

    parser.add_argument("--train_embed", action="store_true",
                      dest="train_embed", default=False)
    parser.add_argument("--train_partial", action="store_true",
                      dest="train_partial", default=False)
    parser.add_argument("--full_tree", dest="full_tree",
                      default='# conversion_status = complete')
    parser.add_argument("--partial_tree", dest="partial_tree",
                      default='# conversion_status = no_tree')

    parser.add_argument(
        "--features", dest="features",
        help="Which features to use: form, lemma, upostag, xpostag, feats, char",
        default=['form', 'char'], nargs='+',
        choices=['form', 'lemma', 'upostag', 'xpostag', 'feats', 'char'])
    parser.add_argument(
        "--targets", dest="targets", nargs="+",
        choices=['head', 'deprel', 'lemma', 'upostag', 'xpostag', 'feats',
                 'sent', 'semrel'],
        help="Which targets to predict: head, deprel, lemma, upostag, xpostag, "
             "feats, sent, semrel",
        default=['head', 'deprel', 'lemma', 'upostag', 'feats'])
    parser.add_argument(
        "--loss_weights", type=float, dest="loss_weights", nargs='+',
        help="Importance of each loss", default=[0.2, 0.8, 0.05, 0.05, 0.2])

    parser.add_argument("--form_embed", type=int, dest="form_embed", default=100)
    parser.add_argument("--pos_embed", type=int, dest="pos_embed", default=32)
    parser.add_argument("--xpos_embed", type=int, dest="xpos_embed", default=32)
    parser.add_argument("--feat_embed", type=int, dest="feat_embed", default=32)
    parser.add_argument("--char_embed", type=int, dest="char_embed", default=64)
    parser.add_argument("--char_max_len", type=int, dest="char_max_len", default=30)

    parser.add_argument("--lstm_layers", type=int, dest="lstm_layers", default=2)
    parser.add_argument("--lstm_hidden_size", type=int, dest="lstm_hidden_size", default=512)
    parser.add_argument("--lstm_dropout", type=float, dest="lstm_dropout", default=0.25)
    parser.add_argument("--head_hidden_size", type=int, dest="head_hidden_size", default=512)
    parser.add_argument("--deprel_hidden_size", type=int, dest="deprel_hidden_size", default=128)
    parser.add_argument("--lemma_hidden_size", type=int, dest="lemma_hidden_size", default=256)
    parser.add_argument("--pos_hidden_size", type=int, dest="pos_hidden_size", default=64)
    parser.add_argument("--xpos_hidden_size", type=int, dest="xpos_hidden_size", default=128)
    parser.add_argument("--feat_hidden_size", type=int, dest="feat_hidden_size", default=128)
    parser.add_argument("--semrel_hidden_size", type=int, dest="semrel_hidden_size", default=64)
    parser.add_argument("--dense_dropout", type=float, dest="dense_droput", default=0.25)
    parser.add_argument("--input_dropout", type=float, dest="input_droput", default=0.25)

    parser.add_argument("--batch_size", type=int, dest="batch_size", default=2500)
    parser.add_argument("--epochs", type=int, dest="epochs", default=400)
    parser.add_argument("--lr", type=float, dest="learning_rate", default=0.002)
    parser.add_argument("--cycle_loss_n", type=int, dest="cycle_loss_n", default=3)
    parser.add_argument("--cycle_loss_weight", type=float, dest="cycle_loss_weight", default=1.0)
    parser.add_argument("--verbose", type=int, dest="verbose", default=1)

    parser.add_argument("--force_trees", action="store_true", dest="force_trees", default=False)
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", default=False)
    parser.add_argument("--continue", action="store_true", dest="continue_training", default=False)
    parser.add_argument("--lower", action="store_true", dest="lower", default=False)
    parser.add_argument("--freeze", action="store_true", dest="freeze", default=False)
    parser.add_argument("--save_probs", action="store_true", dest="save_probs", default=False)
    parser.add_argument("--reload_params", action="store_true", dest="reload_params", default=False)

    params = parser.parse_args()
    valid_params(params)

    if 'semrel' in params.targets:
        loader = ConllSemanticLoader()
        saver = ConllSemanticSaver()
    else:
        loader = ConllLoader()
        saver = ConllSaver()

    if params.mode == 'train':
        print('Load train data', time.strftime("%Y-%m-%d %H:%M:%S"))
        train_data = loader.load(params.train)

        print('Start training', time.strftime("%Y-%m-%d %H:%M:%S"))
        if params.continue_training:
            print('Load model', time.strftime("%Y-%m-%d %H:%M:%S"))
            parser = joblib.load(params.model_file)

            if params.reload_params:
                parser.params = params
                w = parser.model.get_weights()
                parser.model = parser.create()
                parser.model.set_weights(w)
                del w
        else:
            print('Initiate model', time.strftime("%Y-%m-%d %H:%M:%S"))
            parser = Parser(params)

        parser.fit(train_data)
        parser.model.summary()

        print('Save model', time.strftime("%Y-%m-%d %H:%M:%S"))
        joblib.dump(parser, params.model_file)

        if params.evaluate:
            print('Predict on train', time.strftime("%Y-%m-%d %H:%M:%S"))
            pred_train = parser.predict(train_data)
            print_summary(pred_train, train_data)

            if params.valid is not None:
                print('Load valid data', time.strftime("%Y-%m-%d %H:%M:%S"))
                valid_data = loader.load(params.valid)

                print('Predict on valid', time.strftime("%Y-%m-%d %H:%M:%S"))
                pred_valid = parser.predict(valid_data)
                print_summary(pred_valid, valid_data)

    elif params.mode == 'multitrain':
        if params.evaluate and params.valid is not None:
            print('Load valid data', time.strftime("%Y-%m-%d %H:%M:%S"))
            valid_data = loader.load(params.valid)

        if params.continue_training:
            print('Load model', time.strftime("%Y-%m-%d %H:%M:%S"))
            parser = joblib.load(params.model_file)

            if params.reload_params:
                parser.params = params
                w = parser.model.get_weights()
                parser.model = parser.create()
                parser.model.set_weights(w)
                del w
        else:
            print('Initiate model', time.strftime("%Y-%m-%d %H:%M:%S"))
            parser = Parser(params)

        print('Start training', time.strftime("%Y-%m-%d %H:%M:%S"))
        for file_name in os.listdir(params.train):
            if 'conll' not in file_name:
                continue

            print('Load train data:', file_name, time.strftime("%Y-%m-%d %H:%M:%S"))
            train_data = loader.load(params.train + file_name)
            parser.fit(train_data)

            if params.evaluate and params.valid is not None:
                print('Predict on valid', time.strftime("%Y-%m-%d %H:%M:%S"))
                pred_valid = parser.predict(valid_data)
                print_summary(pred_valid, valid_data)

        print('Save model', time.strftime("%Y-%m-%d %H:%M:%S"))
        joblib.dump(parser, params.model_file)

        if params.evaluate:
            print('Predict on train', time.strftime("%Y-%m-%d %H:%M:%S"))
            pred_train = parser.predict(train_data)
            print_summary(pred_train, train_data)

            if params.valid is not None:
                print('Predict on valid', time.strftime("%Y-%m-%d %H:%M:%S"))
                pred_valid = parser.predict(valid_data)
                print_summary(pred_valid, valid_data)

    elif params.mode == 'autotrain':
        print('Load train data', time.strftime("%Y-%m-%d %H:%M:%S"))
        train_data = loader.load(params.train)

        print('Load valid data', time.strftime("%Y-%m-%d %H:%M:%S"))
        valid_data = loader.load(params.valid)

        print('Start training', time.strftime("%Y-%m-%d %H:%M:%S"))
        print('Initiate model', time.strftime("%Y-%m-%d %H:%M:%S"))
        threshold = 0.001
        decreases = 2
        patience = 5
        decrease_factor = 2
        epochs_per_eval = 5
        iters = params.epochs//epochs_per_eval

        if params.continue_training:
            print('Load model', time.strftime("%Y-%m-%d %H:%M:%S"))
            parser = joblib.load(params.model_file)

            if params.reload_params:
                parser.params = params
                w = parser.model.get_weights()
                parser.model = parser.create()
                parser.model.set_weights(w)
                del w
        else:
            print('Initiate model', time.strftime("%Y-%m-%d %H:%M:%S"))
            parser = Parser(params)
            
        parser.params.epochs = epochs_per_eval

        scores = []
        score_func = full_score
        for i in range(iters):
            print('Train iter', i, time.strftime("%Y-%m-%d %H:%M:%S"))
            parser.fit(train_data)
            pred_valid = parser.predict(valid_data)
            score = score_func(pred_valid, valid_data)

            print_summary(pred_valid, valid_data)
            print('summary', patience, decreases, threshold, score, time.strftime("%Y-%m-%d %H:%M:%S"))

            if len(scores) and score - max(scores) < threshold:
                if patience == 0:
                    if decreases > 0:
                        prev_lr = K.get_value(parser.model.optimizer.lr)
                        K.set_value(parser.model.optimizer.lr, prev_lr/decrease_factor)
                        print('lr change', prev_lr, K.get_value(parser.model.optimizer.lr))
                        threshold /= decrease_factor
                        decreases -= 1
                        patience = 5
                    else:
                        break
                else:
                    patience -= 1
            else:
                patience = 5
            
            scores.append(score)

        print('Finished training', time.strftime("%Y-%m-%d %H:%M:%S"))
        parser.model.summary()
        print(scores)

        print('Save model', time.strftime("%Y-%m-%d %H:%M:%S"))
        joblib.dump(parser, params.model_file)

        if params.evaluate:
            print('Predict on train', time.strftime("%Y-%m-%d %H:%M:%S"))
            pred_train = parser.predict(train_data)
            print_summary(pred_train, train_data)

            if params.valid is not None:
                print('Predict on valid', time.strftime("%Y-%m-%d %H:%M:%S"))
                pred_valid = parser.predict(valid_data)
                print_summary(pred_valid, valid_data)

    elif params.mode == 'evaluate':
        print('Load model', time.strftime("%Y-%m-%d %H:%M:%S"))
        parser = joblib.load(params.model_file)
        if params.reload_params:
            parser.params = params
            w = parser.model.get_weights()
            parser.model = parser.create()
            parser.model.set_weights(w)
            del w

        print('Load data', time.strftime("%Y-%m-%d %H:%M:%S"))
        test_data = loader.load(params.test)

        print('Predict', time.strftime("%Y-%m-%d %H:%M:%S"))
        pred = parser.predict(test_data)

        print_summary(pred, test_data)

    elif params.mode == 'predict':
        print('Load model', time.strftime("%Y-%m-%d %H:%M:%S"))
        parser = joblib.load(params.model_file)
        if params.reload_params:
            parser.params = params
            w = parser.model.get_weights()
            parser.model = parser.create()
            parser.model.set_weights(w)
            del w

        print('Load data', time.strftime("%Y-%m-%d %H:%M:%S"))
        test_data = loader.load(params.test)

        print('Predict', time.strftime("%Y-%m-%d %H:%M:%S"))
        pred = parser.predict(test_data)

        print('Save predictions', time.strftime("%Y-%m-%d %H:%M:%S"))
        saver.save(params.pred_file, pred)

    elif params.mode == 'multipredict':
        print('Load model', time.strftime("%Y-%m-%d %H:%M:%S"))
        parser = joblib.load(params.model_file)
        if params.reload_params:
            parser.params = params
            w = parser.model.get_weights()
            parser.model = parser.create()
            parser.model.set_weights(w)
            del w

        for file_name in os.listdir(params.test):
            if 'conll' not in file_name:
                continue

            print('Load data:', file_name, time.strftime("%Y-%m-%d %H:%M:%S"))
            test_data = loader.load(params.test + file_name)

            print('Predict:', file_name, time.strftime("%Y-%m-%d %H:%M:%S"))
            pred = parser.predict(test_data)

            print('Save predictions:', file_name, time.strftime("%Y-%m-%d %H:%M:%S"))
            saver.save(params.pred_file + file_name, pred)

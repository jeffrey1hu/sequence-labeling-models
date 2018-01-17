import argparse
import logging
import sys
from datetime import datetime

import numpy as np

from utils.data_util import get_chunks
from utils.data_util import read_conll, HmmModelHelper
from utils.defs import LBLS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_word_features = 2 # Number of features for every word in the input.
    window_size = 1
    n_features = (2 * window_size + 1) * n_word_features # Number of features for every word in the input.

    def __init__(self, args):
        self.model = "hmm"

        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.model, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format(self.model)
        self.log_output = self.output_path + "log"


class HmmModel(object):
    def __init__(self, tok2id, lbs):

        self.tok2id = tok2id
        self.labels = lbs

        # will hold conditional frequency distribution for P(Ci+1|Ci)
        self.A = np.zeros((len(lbs), len(lbs)))

        # will hold conditional frequency distribution for P(Wi|Ck)
        self.B = np.zeros((len(lbs), len(tok2id)))

        self.pi = np.zeros((len(lbs)))

        pass

    def train(self, sentences):
        counter = 0
        for tokens, labels in sentences:
            for idx, token in enumerate(tokens):
                current_label = labels[idx]
                self.B[current_label, token] += 1
                if idx == 0:
                    self.pi[current_label] += 1
                else:
                    self.A[labels[idx-1], current_label] += 1
            counter += 1
            if counter % 5000 == 0:
                print "process %dth sentence" % counter
        self.A = self.A / self.A.sum(axis=1).reshape((-1, 1))
        self.B = self.B / self.B.sum(axis=1).reshape((-1, 1))
        self.pi = self.pi / self.pi.sum()

    def viterbi_decoder(self, seqs):
        prob_matrix = np.zeros((len(seqs), len(self.labels)))
        path_matrix = np.zeros((len(seqs), len(self.labels)))
        for idx, token in enumerate(seqs):
            if idx == 0:
                prob_matrix[0] = self.pi * self.B[:, token]
                continue
            for i in range(len(self.labels)):
                for j in range(len(self.labels)):
                    # j previous label idx, i current label idx
                    temp_prob = prob_matrix[idx-1, j] * self.A[j, i]
                    if temp_prob > prob_matrix[idx, i]:
                        prob_matrix[idx, i] = temp_prob
                        path_matrix[idx, i] = j
        pathes = [prob_matrix[-1, :].argmax()]
        for k in range(len(seqs)-1, 0, -1):
            pathes.append(int(path_matrix[k, pathes[-1]]))
        pathes.reverse()
        return pathes

    def predict(self, batch_data):
        result = []
        for seqs, labels in batch_data:
            result.append(self.viterbi_decoder(seqs))
        return result


def evaluate(model, examples):

    correct_preds, total_correct, total_preds = 0., 0., 0.
    predicts = model.predict(examples)
    labels = map(lambda x: x[1], examples)

    for predict, label in zip(predicts, labels):
        gold = set(get_chunks(label))
        pred = set(get_chunks(predict))
        correct_preds += len(gold.intersection(pred))
        total_preds += len(pred)
        total_correct += len(gold)
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return (p, r, f1)


def load_and_preprocess_data(args):
    logger.info("Loading training data...")
    train = read_conll(args.data_train)
    logger.info("Done. Read %d sentences", len(train))
    logger.info("Loading dev data...")
    dev = read_conll(args.data_dev)
    logger.info("Done. Read %d sentences", len(dev))

    helper = HmmModelHelper.build(train)

    # now process all the input data.
    train_data = helper.vectorize(train)
    dev_data = helper.vectorize(dev)

    return helper, train_data, dev_data, train, dev


def do_train(args):
    # Set up some parameters.
    config = Config(args)
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)

    helper.save(config.output_path)
    hmm = HmmModel(helper.tok2id, LBLS)
    hmm.train(train)

    p, r, f1 = evaluate(hmm, dev)

    print "p, {}, r, {}, f1 {}".format(p, r, f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/train.conll", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/dev.conll", help="Dev data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.set_defaults(func=do_train)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)

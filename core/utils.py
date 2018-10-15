import numpy as np
import tensorflow as tf
import cPickle as pickle
import json
import time
import heapq
import os
import sys
import logging
sys.path.append('../coco-caption')
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

def load_coco_data(data_path='./data', split='train'):
    data_path = os.path.join(data_path, split)
    start_t = time.time()
    data = {}

    data['split'] = split
    data['features_path'] = os.path.join(data_path, 'feats')
    data['n_examples'] = len(os.listdir(data['features_path']))

    if split == 'train':
        with open(os.path.join(data_path, '%s.captions.pkl' %split), 'rb') as f:
            data['captions'] = pickle.load(f)
        with open(os.path.join(data_path, 'word_to_idx.pkl'), 'rb') as f:
            data['word_to_idx'] = pickle.load(f)

    anno_path = os.path.join(data_path, '%s.annotations.pkl' % (split))
    annotations = load_pickle(anno_path)
    data['image_id'] = annotations['image_id'].as_matrix()
    data['file_name'] = annotations['file_name'].as_matrix()

    for k, v in data.iteritems():
        if type(v) == np.ndarray:
            print k, type(v), v.shape, v.dtype
        elif type(v) != int:
            print k, type(v), len(v)
        else:
            print k, type(v), v

    end_t = time.time()
    print "Elapse time: %.2f" %(end_t - start_t)
    return data


def decode_captions(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == '<END>':
                words.append('.')
                break
            if word != '<NULL>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded


def sample_coco_minibatch(data, batch_size):
    data_size = data['n_examples']
    mask = np.random.choice(data_size, batch_size)
    file_names = data['file_name'][mask]
    return mask, file_names


def write_bleu(scores, path, epoch, iteration):
    with open(os.path.join(path, 'val.bleu.scores.txt'), 'a') as f:
        f.write('Epoch %d. Iteration %d\n' %(epoch+1, iteration+1))
        f.write('Bleu_1: %f\n' %scores['Bleu_1'])
        f.write('Bleu_2: %f\n' %scores['Bleu_2'])
        f.write('Bleu_3: %f\n' %scores['Bleu_3'])
        f.write('Bleu_4: %f\n' %scores['Bleu_4'])
        f.write('METEOR: %f\n' %scores['METEOR'])
        f.write('ROUGE_L: %f\n' %scores['ROUGE_L'])
        f.write('CIDEr: %f\n\n' %scores['CIDEr'])


def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' %path)
        return file


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)
        print ('Saved %s..' % path)

def evaluate(data_path='./data', split='val', get_scores=False):
    reference_path = os.path.join(data_path, "annotations/captions_%s2017.json" %(split))
    candidate_path = os.path.join(data_path, "%s/%s.candidate.captions.json" %(split, split))

    # load caption data
    ref = COCO(reference_path)
    hypo = ref.loadRes(candidate_path)

    cocoEval = COCOEvalCap(ref, hypo)
    cocoEval.evaluate()
    final_scores = {}
    for metric, score in cocoEval.eval.items():
        final_scores[metric] = score
        logging.info('%s:\t%.3f'%(metric, score))

    if get_scores:
        return final_scores

def define_logger(logging_level='info', log_file=None):
        logging_level = logging.INFO if logging_level.lower() == 'info' \
                   else logging.WARNING if logging_level.lower() == 'warning' \
                   else logging.DEBUG

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Set logging to console
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s]|%(asctime)s| - %(message)s")
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
        # Set logging to file
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s]|%(asctime)s| - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

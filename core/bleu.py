import cPickle as pickle
import json
import os
import sys
sys.path.append('../coco-caption')
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


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
        print '%s:\t%.3f'%(metric, score)

    if get_scores:
        return final_scores
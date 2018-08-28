from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
import torch
from torchvision import datasets
from torchvision import transforms
from core.utils import *
from feature_extractor import FeatureExtractor
from feature_extractor import CocoDataset

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm


def _process_caption_data(split, caption_file=None, image_dir=None, max_length=None):
    if split == 'test' or split == 'val':
        file_paths = []
        for dirpath, dirnames, filenames in os.walk(image_dir):
            for filename in filenames:
                file_paths.append(os.path.join(dirpath, filename))

        data = [{'image_id':image_names.split('/')[-1].split('.')[0].lstrip('0'), 'file_name':image_names}
                    for image_names in file_paths]
        
        caption_data = pd.DataFrame.from_dict(data)
        caption_data.sort_values(by='image_id', inplace=True)
        caption_data = caption_data.reset_index(drop=True)
        caption_data['id'] = caption_data.index
        return caption_data

    with open(caption_file) as f:
        caption_data = json.load(f)

    # id_to_filename is a dictionary such as {image_id: filename]} 
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
    data = []
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id])
        data += [annotation]
    
    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data)
    del caption_data['id']
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)
    
    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        caption = " ".join(caption.split())  # replace multiple spaces
        
        caption_data.set_value(i, 'caption', caption.lower())
        if max_length != None and len(caption.split(" ")) > max_length:
            del_idx.append(i)
    
    # delete captions if size is larger than max_length
    print "The number of captions before deletion: %d" %len(caption_data)
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print "The number of captions after deletion: %d" %len(caption_data)
    return caption_data


def _build_vocab(annotations, threshold=1, max_size=None):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] +=1
        
        if len(caption.split(' ')) > max_len:
            max_len = len(caption.split(' '))

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1

    print "Max length of caption: ", max_len
    return word_to_idx


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)   

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
            else:
                cap_vec.append(word_to_idx['NULL'])
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 
        
        captions[i, :] = np.asarray(cap_vec)
    print "Finished building caption vectors"
    return captions


def main():
    # batch size for extracting feature vectors from vggnet.
    batch_size = 100
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.  
    max_length = 30
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1
    
    datasets = {}
    for split in splits:
        datasets[split] = _process_caption_data(split,
                                                caption_file='data/annotations/captions_train2017.json',
                                                image_dir='image/train/',
                                                max_length=max_length)
        save_pickle(datasets[split], 'data/%s/%s.annotations.pkl' % (split, split))

    print 'Finished processing caption data'

    annotations = load_pickle('./data/train/train.annotations.pkl'

    word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold)
    save_pickle(word_to_idx, './data/train/word_to_idx.pkl')
    
    captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
    save_pickle(captions, './data/train/train.captions.pkl')

    # extract conv5_3 feature vectors
    feature_extractor = FeatureExtractor(model_name='resnet101', layer=3)
    for split in splits:
        if not os.path.isdir('./data/%s/feats/' % (split)):
            os.mkdir('./data/%s/feats/' % (split))
        anno_path = './data/%s/%s.annotations.pkl' % (split, split)
        annotations = load_pickle(anno_path)
        file_names = list(annotations['file_name'].unique())
        dataset = CocoDataset(file_names=file_names)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)

        for i, (batch_ids, batch_images) in enumerate(tqdm(data_loader)):
            feats = feature_extractor(batch_image).data.cpu().numpy()
            feats = feats.reshape(-1, feats.shape[1]*feats.shape[2], feats.shape[-1])
            for j in range(len(feats)):
                np.save('./data/%s/feats/%d.npy' % (split, batch_ids[j]), feats[j])

if __name__ == "__main__":
    main()

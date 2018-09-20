from collections import Counter
import torch
from torchvision import datasets
from torchvision import transforms
from core.utils import *
from feature_extractor import FeatureExtractor
from feature_extractor import CocoDataset

from tensorflow import flags
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm

"""Parameters for pre-processing"""
FLAGS = flags.FLAGS

flags.DEFINE_string('phases', 'train,val,test', 'Phases in which you want to pre-process the dataset. '+
                                                'Phases should be seperated by commas and no space. '+
                                                'Images of phase named <phase> should be placed in image/<phase>. '+
                                                'Default is pre-process all splits of COCO-dataset.')

flags.DEFINE_integer('batch_size', 128,         'Batch size to be used for extracting features from images.')

flags.DEFINE_integer('max_length', 30,          'Max length, only be used to pre-process captions in training split of dataset. '+
                                                'Captions have more words than max_length will be removed.')

flags.DEFINE_integer('word_count_threshold', 1, 'Words occur less than word_count_threshold times in the dataset '+
                                                '(only apply for training set) will be removed from vocabulary '+
                                                'and replaced by <UNK> token in the captions.')

flags.DEFINE_integer('vocab_size', 0,           'Size of vocabulary. '+
                                                'Vocabulary is made of vocab_size most frequent words in the dataset. '+
                                                'Leave it to default value means not using it.')

flags.DEFINE_string('model_name', 'resnet', 'Model used to extract features of images.'+
                                                'It should be vgg or resnet.')

flags.DEFINE_string('model_num_layers', '101',  'Number of layers for model\'s architecture.'+
                                                'If model_name is vgg, this variable can take values of 16 or 19.'+
                                                'If model_name is resnet, this variable can take values of 50, 101 or 152.')

flags.DEFINE_boolean('use_tf', False, 'Whether to use Tensorflow slim model to extract features of images or use pytorch.'+
                                                'Using Pytorch will be faster but it will keep you from finetuning the extracting model.'+
                                                'Default is set to use Pytorch models.')

flags.DEFINE_string('model_ckpt', '', 'Model checkpoint to load model\'s weights to extract images.'+
                                                'Only use this variable when you use tensorflow model.')

def _process_caption_data(phase, caption_file=None, image_dir=None, max_length=None):
    if phase == 'test' or phase == 'val':
        if phase == 'val':
            with open(caption_file, 'r') as f:
                caption_object = json.load(f)
            caption_object['type'] = 'caption'

        file_paths = []
        for dirpath, dirnames, filenames in os.walk(image_dir):
            for filename in filenames:
                file_paths.append(os.path.join(dirpath, filename))

        data = [{'image_id':int(image_name.split('/')[-1].split('_')[-1].split('.')[0].lstrip('0')), 'file_name':image_name}
                    for image_name in file_paths]
        
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
        caption = caption.replace('.','').replace(',','').replace("'",'').replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(')','').replace('-',' ')
        caption = ' '.join(caption.split())  # replace multiple spaces
        
        caption_data.set_value(i, 'caption', caption.lower())
        if max_length != None and len(caption.split(' ')) > max_length:
            del_idx.append(i)
    
    # delete captions if size is larger than max_length
    print "The number of captions before deletion: %d" %len(caption_data)
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print "The number of captions after deletion: %d" %len(caption_data)
    return caption_data


def _build_vocab(annotations, threshold=1, vocab_size=0):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] += 1
        
        if len(caption.split(' ')) > max_len:
            max_len = len(caption.split(' '))

    if vocab_size > 0:
        top_n_counter = [w for w, n in counter.most_common(vocab_size)]
        vocab = [word for word in counter if counter[word] >= threshold and word in top_n_counter]
    else:
        vocab = [word for word in counter if counter[word] >= threshold]

    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2, u'<UNK>': 3}
    idx = len(word_to_idx)
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
                cap_vec.append(word_to_idx['<UNK>'])
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 
        
        captions[i, :] = np.asarray(cap_vec)
    print "Finished building caption vectors"
    return captions

def main():
    # phases to be processed.
    phases = FLAGS.phases.split(',')
    # batch size for extracting feature vectors.
    batch_size = FLAGS.batch_size 
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.  
    max_length = FLAGS.max_length
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = FLAGS.word_count_threshold
    vocab_size = FLAGS.vocab_size

    model_name = FLAGS.model_name + FLAGS.model_num_layers
    
    datasets = {}
    for phase in phases:
        datasets[phase] = _process_caption_data(phase,
                                                caption_file='data/annotations/captions_%s2017.json' % phase,
                                                image_dir='image/%s/' % phase,
                                                max_length=max_length)
        save_pickle(datasets[phase], 'data/%s/%s.annotations.pkl' % (phase, phase))

    print 'Finished processing caption data'

    annotations = load_pickle('./data/train/train.annotations.pkl')

    word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold, vocab_size=vocab_size)
    save_pickle(word_to_idx, './data/train/word_to_idx.pkl')
    
    captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
    save_pickle(captions, './data/train/train.captions.pkl')

    if FLAGS.use_tf:
        tf_datasets = TensorFlowCocoDataset(phases)
        feature_extractor = TensorFlowFeatureExtracter(FLAGS.model_name, FLAGS.model_num_layers, FLAGS.model_ckpt)
    else:
        feature_extractor = FeatureExtractor(model_name=model_name, layer=3)

    for phase in phases:
        if not os.path.isdir('./data/%s/feats/' % (phase)):
            os.mkdir('./data/%s/feats/' % (phase))

        if FLAGS.use_tf:
            datasets_iter = tf_datasets.get_iter()
            features = feature_extractor.get_features(datasets_iter)
            with tf.Session() as sess:
                saver = tf.train.Saver()
                saver.restore(sess, model_ckpt)
                    phase_init_op = datasets_iter.make_initializer(tf_datasets[phase][0])
                    image_ids = tf_datasets[phase][1]
                    sess.run(phase_init_op)
                    for i in tqdm(range(len(image_ids) // batch_size + 1)):
                            feature_vals = sess.run(features)
                            feature_vals = feature_vals.reshape(-1, feature_vals.shape[1]*feature_vals.shape[2], feature_vals.shape[-1])
                            for j in range(len(feature_vals)):
                                np.save('./data/%s/feats/%d.npy' % (split, image_ids[i*batch_size+j]), feature_vals[j])

        else:
            anno_path = './data/%s/%s.annotations.pkl' % (phase, phase)
            annotations = load_pickle(anno_path)
            file_names = list(annotations['file_name'].unique())
            dataset = CocoDataset(file_names=file_names)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)

            for i, (batch_ids, batch_images) in enumerate(tqdm(data_loader)):
                feats = feature_extractor(batch_images).data.cpu().numpy()
                feats = feats.reshape(-1, feats.shape[1]*feats.shape[2], feats.shape[-1])
                for j in range(len(feats)):
                    np.save('./data/%s/feats/%d.npy' % (phase, batch_ids[j]), feats[j])
        

if __name__ == "__main__":
    main()

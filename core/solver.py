import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os
import cPickle as pickle
from scipy import ndimage
from scipy.misc import imresize
from utils import *
from tqdm import tqdm
import logging

class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17)
                - image_idxs: Indices for mapping caption to image of shape (400000, )
                - word_to_idx: Mapping dictionary from word to index
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_model: String; model path for test
        """

        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_score = kwargs.pop('print_score', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.eval_every = kwargs.pop('eval_every', 200)
        self.save_every = kwargs.pop('save_every', 200)
        self.start_from = kwargs.pop('start_from', None)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def _read_features(self, data, ids_list):
        batch_feats = np.array([np.load(os.path.join(data['features_path'], str(ids) + '.npy')) for ids in ids_list])
        return batch_feats

    def train(self, beam_size=1):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Set logging to console
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s]|%(asctime)s| - %(message)s")
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
        # Set logging to file
        handler = logging.FileHandler('train.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s]|%(asctime)s| - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)


        # train/val dataset
        # Changed this because I keep less features than captions, see prepro
        n_examples = self.data['n_examples']
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
        captions = self.data['captions']
        image_ids = self.data['image_id']
        n_iters_val = int(np.ceil(float(self.val_data['n_examples'])/self.batch_size))

        # build graphs for training model and sampling captions
        # This scope fixed things!!
        with tf.variable_scope(tf.get_variable_scope()):
            loss = self.model.build_model()
            tf.get_variable_scope().reuse_variables()
            if beam_size == 1:
                model_sampler_ops = self.model.build_sampler(max_len=35)
            else:
                model_sampler_ops = self.model.build_sampler_with_beam_search(max_len=35, beam_size=beam_size)
        # train op
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = self.optimizer(learning_rate=self.learning_rate, momentum=0.9)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

        # summary op
        # tf.scalar_summary('batch_loss', loss)
        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            #tf.histogram_summary(var.op.name, var)
            tf.summary.histogram(var.op.name, var)
        for grad, var in grads_and_vars:
            #tf.histogram_summary(var.op.name+'/gradient', grad)
            tf.summary.histogram(var.op.name+'/gradient', grad)

        #summary_op = tf.merge_all_summaries()
        summary_op = tf.summary.merge_all()

        # Summary for BLEU-1 score
        bleu_score = tf.placeholder(tf.float32, [])
        bleu_summary_op = tf.summary.scalar('Bleu-1 Score', bleu_score)

        logging.info("The number of epoch: %d" %self.n_epochs)
        logging.info("Data size: %d" %n_examples)
        logging.info("Batch size: %d" %self.batch_size)
        logging.info("Iterations per epoch: %d" %n_iters_per_epoch)

        config = tf.ConfigProto(allow_soft_placement = True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            #summary_writer = tf.train.SummaryWriter(self.log_path, graph=tf.get_default_graph())
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=20)

            if self.pretrained_model is not None:
                logging.info("Start training with pretrained Model.")
                saver.restore(sess, self.pretrained_model)
            if self.start_from is not None:
                assign_zero_op = global_step.assign(self.start_from)
                sess.run(assign_zero_op)
            gs = sess.run(global_step)
            logging.info("Start training at %d time-step.", gs)
            prev_loss = -1
            max_bleu_score = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_ids = image_ids[rand_idxs]

                for i in range(n_iters_per_epoch):
                    captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                    image_ids_batch = image_ids[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch = self._read_features(self.data, image_ids_batch)
                    feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}
                    _, l, gs = sess.run([train_op, loss, global_step], feed_dict)
                    curr_loss += l

                    # write summary for tensorboard visualization
                    if (gs+1) % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, gs+1)

                    if (gs+1) % self.print_every == 0:
                        logging.info("Train loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, gs+1, l))
                        ground_truths = captions[image_ids == image_ids_batch[0]]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print "Ground truth %d: %s" %(j+1, gt)
                        gen_caps = sess.run(model_sampler_ops[-1], feed_dict) # generated_captions is last element of the model_sampler_ops
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print "Generated caption: %s\n" %decoded[0]


                    # print out BLEU scores and file write
                    if self.print_score and (gs+1) % self.eval_every == 0:
			self.test(self.val_data, split='val', model_sampler_ops=model_sampler_ops, sess=sess)

                        scores = evaluate(data_path='./data', split='val', get_scores=True)
                        write_bleu(scores=scores, path=self.model_path, epoch=e, iteration=gs+1)
                        bleu_summary = sess.run(bleu_summary_op, feed_dict={bleu_score: scores['Bleu_1']})
                        summary_writer.add_summary(bleu_summary, gs+1)
                        if max_bleu_score < scores['Bleu_1']:
                            saver.save(sess, os.path.join(self.model_path, 'best_model'))
                            max_bleu_score = scores['Bleu_1']

                    # save model's parameters
                    if (gs+1) % self.save_every == 0:
                        saver.save(sess, os.path.join(self.model_path, 'model_%d' % (gs+1)), global_step=gs+1)
                        print "model-%s saved." %(gs+1)

                logging.info("Previous epoch loss: ", prev_loss)
                logging.info("Current epoch loss: ", curr_loss)
                logging.info("Elapsed time: ", time.time() - start_t)
                prev_loss = curr_loss
                curr_loss = 0

    def test(self, data, split='train', attention_visualization=False, save_sampled_captions=True, beam_size=1, model_sampler_ops=None, sess=None):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''
        # build a graph to sample captions
	if model_sampler_ops:
	    alphas, betas, sampled_captions = model_sampler_ops
	else:
            if beam_size == 1:
                alphas, betas, sampled_captions = self.model.build_sampler(max_len=35)    # (N, max_len, L), (N, max_len)
            else:
                alphas, betas, sampled_captions = self.model.build_sampler_with_beam_search(max_len=35, beam_size=beam_size)    # (N, max_len, L), (N, max_len)
	if not sess:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                saver = tf.train.Saver()
                saver.restore(sess, self.test_model)

        if attention_visualization:
    	    mask, image_files = sample_coco_minibatch(data, self.batch_size)

	    features_batch = self._read_features(data, data[mask])
	    feed_dict = { self.model.features: features_batch }
	    alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
	    decoded = decode_captions(sam_cap, self.model.idx_to_word)

	    for n in range(len(decoded)):
	        print "Sampled Caption: %s" %decoded[n]
	        fig = plt.figure(figsize=(19.2,10.8), dpi=300)
	        # Plot original image
	        img = ndimage.imread(image_files[n])
	        img = imresize(img, (224, 224))
	        plt.subplot(4, 5, 1)
	        plt.imshow(img)
	        plt.axis('off')

	        # Plot images with attention weights
	        words = decoded[n].split(" ")
	        for t in range(len(words)):
		    if t > 18:
		        break
		    plt.subplot(4, 5, t+2)
		    plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
		    plt.imshow(img)
		    alp_curr = alps[n,t,:].reshape(14,14)
	    	    alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
		    plt.imshow(alp_img, alpha=0.85)
	    	    plt.axis('off')
	        plt.show()
	        fig.savefig('samples/%d.jpg' % n)

        if save_sampled_captions:
            all_sam_cap = np.ndarray((data['n_examples'], 35))
            num_iter = int(np.ceil(float(data['n_examples']) / self.batch_size))
            for i in tqdm(range(num_iter)):
                start = i * self.batch_size
                end = min((i+1) * self.batch_size, self.val_data['n_examples'])
                ids_batch = data['image_id'][start:end]
                features_batch = self._read_features(data, ids_batch)
                feed_dict = { self.model.features: features_batch}
                all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)
            all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
            all_decoded = [{"image_id": int(self.val_data['image_id'][i]), "caption": caption} 
                            for i, caption in enumerate(all_decoded)]
            save_json(all_decoded, "./data/%s/%s.candidate.captions.json" % (split, split))

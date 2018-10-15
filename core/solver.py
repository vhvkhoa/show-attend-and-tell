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
    def __init__(self, model, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_model: String; model path for test
        """

        self.model = model
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.metric = kwargs.pop('metric', 'CIDEr')
        self.print_every = kwargs.pop('print_every', 100)
        self.eval_every = kwargs.pop('eval_every', 200)
        self.start_from = kwargs.pop('start_from', 0)
        self.log_path = kwargs.pop('log_path', './log/')
        self.checkpoint_dir = kwargs.pop('checkpoint_dir', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', '')
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=0.9, decay=0.95)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def _read_features(self, data, ids_list):
        batch_feats = np.array([np.load(os.path.join(data['features_path'], str(ids) + '.npy')) for ids in ids_list])
        return batch_feats

    def train(self, data, val_data, beam_size=1):
        # In addition to printing out INFOs to console, all INFOs would also be saved to file in log_path folder to keep track.
        define_logger(logging_level='info', log_file=os.path.join(self.log_path, 'train_%d.log' % self.start_from))

        # Changed this because I keep less features than captions, see prepro
        n_examples = data['n_examples']
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
        captions = data['captions']
        image_ids = data['image_id']
        n_iters_val = int(np.ceil(float(val_data['n_examples'])/self.batch_size))

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
            optimizer = self.optimizer
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

        # summary op
        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        for grad, var in grads_and_vars:
            tf.summary.histogram(var.op.name+'/gradient', grad)

        summary_op = tf.summary.merge_all()

        # Summary for every metric
        metrics = ['Bleu_%d' % (i+1) for i in range(4)] + ['METEOR', 'ROUGE_L', 'CIDEr']
        score_placeholders = [tf.placeholder(tf.float32, [], metric) for metric in metrics]
        score_summary_op = tf.summary.merge([tf.summary.scalar('%s_score' % metric, score_ph) 
                                                for score_ph, metric in zip(score_placeholders, metrics)])


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
            best_ckpt_saver = tf.train.Saver(max_to_keep=1)

            if self.pretrained_model is not '':
                logging.info("Start training with pretrained Model.")
                saver.restore(sess, self.pretrained_model)
            if self.start_from is not 0:
                assign_global_step_op = global_step.assign(self.start_from)
                sess.run(assign_global_step_op)
            gs = sess.run(global_step)
            logging.info("Start training at %d time-step.", gs)
            prev_loss = -1
            max_bleu_score = -1
            curr_loss = 0
            start_t = time.time()
            max_score = 1000.
            def eval_and_log(current_epoch, current_gs):
                self.test(val_data, split='val', model_sampler_ops=model_sampler_ops, sess=sess)

                scores = evaluate(data_path='./data', split='val', get_scores=True)
                write_bleu(scores=scores, path=self.checkpoint_dir, epoch=current_epoch, iteration=current_gs+1)
                score_summary = sess.run(score_summary_op, feed_dict={score_ph: scores[metric] for score_ph, metric in zip(score_placeholders, metrics)})
                summary_writer.add_summary(score_summary, current_gs+1)

                # save model's parameters
                saver.save(sess, os.path.join(self.checkpoint_dir, 'checkpoint-%d' % (current_gs+1)))
                logging.info("checkpoint-%d saved." %(current_gs+1))

                return scores

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_ids = image_ids[rand_idxs]

                for i in range(n_iters_per_epoch):
                    captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                    image_ids_batch = image_ids[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch = self._read_features(data, image_ids_batch)
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
                    if (gs+1) % self.eval_every == 0:
                        scores = eval_and_log(sess, model_sampler_ops)

                        if max_score < scores[self.metric]:
                            best_ckpt_saver.save(sess, os.path.join(self.checkpoint_dir, 'best_model.ckpt'))
                            max_score = scores[self.metric]

                # eval at the end of epoch to retrain if needed
                eval_and_log(sess, model_sampler_ops)

                logging.info("Previous epoch loss: ", prev_loss)
                logging.info("Current epoch loss: ", curr_loss)
                logging.info("Elapsed time: ", time.time() - start_t)
                prev_loss = curr_loss
                curr_loss = 0

    def test(self, data, attention_visualization=False, save_sampled_captions=True, beam_size=1, model_sampler_ops=None, sess=None):
        # build a graph to sample captions
        if model_sampler_ops:
            alphas, betas, sampled_captions = model_sampler_ops
        else:
            if beam_size == 1:
                alphas, betas, sampled_captions = self.model.build_sampler(max_len=35)
            else:
                alphas, betas, sampled_captions = self.model.build_sampler_with_beam_search(max_len=35, beam_size=beam_size)
        if not sess:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                saver = tf.train.Saver()
                saver.restore(sess, self.test_model)

            if attention_visualization:
                mask, image_files = sample_coco_minibatch(data, 10)

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
                    end = min((i+1) * self.batch_size, data['n_examples'])
                    ids_batch = data['image_id'][start:end]
                    features_batch = self._read_features(data, ids_batch)
                    feed_dict = { self.model.features: features_batch}
                    all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)
                all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
                all_decoded = [{"image_id": int(data['image_id'][i]), "caption": caption} 
                                for i, caption in enumerate(all_decoded)]
                save_json(all_decoded, "./data/%s/%s.candidate.captions.json" % (split, split))

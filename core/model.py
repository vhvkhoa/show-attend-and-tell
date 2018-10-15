# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf
from tf_beam_decoder import beam_decoder
class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_feature=[196, 512], dim_embed=512, dim_hidden=1024, n_time_step=16,
                  prev2out=True, ctx2out=True, alpha_c=0.0, enable_selector=True, dropout=0.5):
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.enable_selector = enable_selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']
        self._end = word_to_idx['<END>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            return context, alpha

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, dropout=0.0, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            h = tf.nn.dropout(h, keep_prob=1.0-dropout)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

    def cell_setup(self, time, *args):
        batch_size = tf.shape(self.args.features)[0]

        c, h = self._get_initial_lstm(features=self.args.features)
        next_cell_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
        self.args.emb_captions = self._word_embedding(inputs=self.captions, reuse=False)

        context, alpha = self._attention_layer(self.args.features, self.args.features_proj, h, reuse=False)
        alpha_ta = tf.TensorArray(tf.float32, self.T + 1)
        alpha_ta = alpha_ta.write(time, alpha)
        if self.enable_selector:
            context, beta = self._selector(context, h, reuse=False)

        next_input = tf.concat([self.args.emb_captions[:,time,:], context], 1)

        loss_ta = tf.TensorArray(tf.float32, size=self.T)
        next_loop_state = (context, alpha_ta, loss_ta)

        emit_output = tf.zeros(self.args.cell_output_size)
        elements_finished = tf.zeros([batch_size], dtype=tf.bool)

        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

    def cell_loop(self, time, cell_output, cell_state, loop_state, *args):
        emit_output = cell_output
        next_cell_state = cell_state
        c, h = cell_state

        past_context, past_alpha_ta, past_loss_ta = loop_state
        
        logits = self._decode_lstm(self.args.emb_captions[:,time-1,:], h, past_context, dropout=self.dropout, reuse=tf.AUTO_REUSE)
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=self.captions[:, time],logits=logits)*self.args.mask[:, time])
        next_loss_ta = past_loss_ta.write(time-1, loss)

        context, alpha = self._attention_layer(self.args.features, self.args.features_proj, h, reuse=True)
        next_alpha_ta = past_alpha_ta.write(time, alpha)
        if self.enable_selector:
            context, beta = self._selector(context, h, reuse=True)

        next_input = tf.concat( [self.args.emb_captions[:,time,:], context], 1)
        next_loop_state = (context, next_alpha_ta, next_loss_ta)

        elements_finished = (time >= self.T)

        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

    def build_model(self):
        features = self.features
        captions = self.captions
        batch_size = tf.shape(features)[0]

        mask = tf.to_float(tf.not_equal(captions, self._null))

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')
        features_proj = self._project_features(features=features)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        cell_output_size = lstm_cell.output_size
        varnames = ['cell_output_size', 'features', 'features_proj', 'mask']
        class Args(object): pass
        self.args = Args()
        for v in varnames:
            setattr(self.args, v, eval(v))

        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:
                return self.cell_setup(time)
            else:
                return self.cell_loop(time, cell_output, cell_state, loop_state)

        emit_ta, final_state, loop_state = tf.nn.raw_rnn(lstm_cell, loop_fn, scope='lstm')
        _, alpha_ta, loss_ta = loop_state
        loss = tf.reduce_sum(loss_ta.stack())
        alphas = tf.transpose(alpha_ta.stack(), (1, 0, 2))[:, :-1, :] # (N, T, L)

        if self.alpha_c > 0:
            alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((float(self.T)/196 - alphas_all) ** 2)
            loss += alpha_reg

        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=20):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        # This scope fixes things:
        with tf.variable_scope('lstm'):
            c, h = self._get_initial_lstm(features=features)
            features_proj = self._project_features(features=features)
        
            for t in range(max_len):
                if t == 0:
                    x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
                else:
                    x = self._word_embedding(inputs=sampled_word, reuse=True)

                context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
                alpha_list.append(alpha)

                if self.enable_selector:
                    context, beta = self._selector(context, h, reuse=(t!=0))
                    beta_list.append(beta)

                with tf.variable_scope('lstm', reuse=(t!=0)):
                    _, (c, h) = lstm_cell(inputs=tf.concat( [x, context],1), state=[c, h])

                logits = self._decode_lstm(x, h, context, reuse=(t!=0))
                sampled_word = tf.argmax(logits, 1)
                sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return alphas, betas, sampled_captions

    def build_sampler_with_beam_search(self, beam_size=10, max_len=20):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')
        features_proj = self._project_features(features=features)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        def tokens_to_inputs_attention_fn(model, symbols, feats, feats_proj, hidden_state, beam_size):
            embed_symbols = model._word_embedding(inputs=tf.reshape(symbols, [-1]), reuse=True)

            context, alpha = self._attention_layer(feats, feats_proj, hidden_state, reuse=True)

            if self.enable_selector:
                context, beta = self._selector(context, hidden_state, reuse=True)

            next_input = tf.concat([embed_symbols, context], 1)
            next_input = tf.reshape(next_input, [-1, beam_size, next_input.shape[-1]])
            return next_input, context, alpha, beta

        def outputs_to_score_attention_fn(model, symbols, outputs, beam_context, beam_size):
            embed_symbols = model._word_embedding(inputs=symbols, reuse=True)
            outputs = tf.reshape(outputs, [-1, outputs.shape[-1]])

            logits = model._decode_lstm(embed_symbols, outputs, beam_context)
            logits = tf.reshape(logits, [-1, beam_size, logits.shape[-1]])
            return tf.nn.log_softmax(logits)

        sampled_captions, logprobs, alphas, betas = beam_decoder(lstm_cell, beam_size, self._start, self._end, 
                                                tokens_to_inputs_attention_fn, outputs_to_score_attention_fn,
                                                features=features, features_proj=features_proj,
                                                max_len=35, selector=self.enable_selector, output_dense=True, scope='lstm', model=self)

        return alphas, betas, sampled_captions

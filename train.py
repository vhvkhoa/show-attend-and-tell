from tensorflow import flags
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data


FLAGS = flags.FLAGS

"""Model's parameters"""
flags.DEFINE_integer('image_feature_size', 196, 'Multiplication of width and height of image feature\'s dimension, e.g 14x14=196 in the original paper.')
flags.DEFINE_integer('image_feature_depth', 1024, 'Depth dimension of image feature, e.g 512 if you extract features at conv-5 of VGG-16 model.')
flags.DEFINE_integer('lstm_hidden_size', 1536, 'Hidden layer size for LSTM cell.')
flags.DEFINE_integer('time_steps', 31, 'Number of time steps to be iterating through.')
flags.DEFINE_integer('embed_dim', 512, 'Embedding space size for embedding tokens.')
flags.DEFINE_integer('beam_size', 3, 'Beam size for inference phase.')
flags.DEFINE_float('dropout', 0.5, 'Dropout portion.')
flags.DEFINE_boolean('prev2out', True, 'Link previous hidden state to output.')
flags.DEFINE_boolean('ctx2out', True, 'Link context features to output.')
flags.DEFINE_boolean('enable_selector', True, 'Enable selector to determine how much important the image context is at every time step.')

"""Training parameters"""
flags.DEFINE_string('optimizer', 'rmsprop', 'Optimizer used to update model\'s weights.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 128, 'Number of examples per mini-batch.')
flags.DEFINE_integer('snapshot_steps', 10, 'Logging every snapshot_steps steps.')
flags.DEFINE_integer('eval_steps', 100, 'Evaluate and save current model every eval_steps steps.')
flags.DEFINE_string('metric', 'CIDEr', 'Metric being based on to choose best model, please insert on of these strings: [Bleu_i, METEOR, ROUGE_L, CIDEr] with i is 1 through 4.')
flags.DEFINE_string('pretrained_model', '', 'Path to a pretrained model to initiate weights from.') 
flags.DEFINE_integer('start_from', 0, 'Step number to start model from, this parameter helps to continue logging in tensorboard from the previous stopped training phase.') 
flags.DEFINE_string('checkpoint_dir', 'checkpoint/', 'Path to directory where checkpoints saved every eval_steps.')
flags.DEFINE_string('log_path', 'log/', 'Path to directory where logs saved during the training process. You can use tensorboard to visualize logging informations and re-read IFO printed on the console in .log files.')

def main():
    # load train dataset
    data = load_coco_data(data_path='./data', split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out scores every epoch
    val_data = load_coco_data(data_path='./data', split='val')

    model = CaptionGenerator(word_to_idx, dim_feature=[FLAGS.image_feature_size, FLAGS.image_feature_depth], dim_embed=FLAGS.embed_dim,
                                    dim_hidden=FLAGS.lstm_hidden_size, n_time_step=FLAGS.time_steps, prev2out=FLAGS.prev2out,
                                    ctx2out=FLAGS.ctx2out, alpha_c=1.0, enable_selector=FLAGS.enable_selector, dropout=FLAGS.dropout)

    solver = CaptioningSolver(model, n_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size, update_rule=FLAGS.optimizer,
                                    learning_rate=FLAGS.learning_rate, metric=FLAGS.metric,
                                    print_every=FLAGS.snapshot_steps, eval_every=FLAGS.eval_steps,
                                    pretrained_model=FLAGS.pretrained_model, start_from=FLAGS.start_from, checkpoint_dir=FLAGS.checkpoint_dir, 
                                    log_path=FLAGS.log_path)

    solver.train(data, val_data, beam_size=FLAGS.beam_size)

if __name__ == "__main__":
    main()

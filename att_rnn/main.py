import tensorflow as tf
import numpy as np
import os
import time
from att_rnn import Attention_RNN
from tensorflow.contrib import learn
from utils import load_embedding, load_data_and_labels, batch_iter, load_test_data, listed_classes
import datetime
import codecs
import pandas as pd
import sys

tf.flags.DEFINE_boolean("is_train", True, "use the train or test function")

# Parameters
tf.flags.DEFINE_float('dev_sample_percentage', .1, "percentage of the training data to use for validation")
tf.flags.DEFINE_string('data_file', '../data/train.txt', "data source for the file")
tf.flags.DEFINE_string('embd_file', '../data/glove.840B.300d.txt.crp', 'embedding file to load')

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default:128)")
tf.flags.DEFINE_integer("sequence_length", 100, "maximum sequence length (default:200)")
tf.flags.DEFINE_string('filter_sizes', '2,3,4,5,7', 'Comma-separated filter sizes (default: "3,4,5")')
tf.flags.DEFINE_integer('num_filters', 10, 'Number of filters per filer size')
tf.flags.DEFINE_float('dropout_keep_prob', 0.9, 'Dropout keep probability (default: 0.5)')

# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 20, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_string("checkpoint_dir", "", "checkpoint directory from training run")
tf.flags.DEFINE_integer('num_checkpoints', 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, 'Log placement of ops on devices')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
print("")
base_dir = '../data/'
vocab_file = base_dir + "vocabulary.pkl"

VOCABULARY_SIZE = 100000 + 1


def test():
    x_test = load_test_data(FLAGS.data_file, vocab_file)
    embedding_matrix = load_embedding(FLAGS.embd_file)
    print("Evaluating...")
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        sess_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)

        sess = tf.Session(config=sess_conf)
        with sess.as_default():
            # load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout").outputs[0]
            embedding_placeholder = graph.get_operation_by_name("embedding/embedding_placeholder").outputs[0]

            # Tensors we want to evaluate
            sigmoid_prob = graph.get_operation_by_name("output/sigmoidprob").outputs[0]

            # Generate batches for one epoch
            batches = batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for idx, x_text_batch in enumerate(batches):
                batch_sigmoid_probs = sess.run(sigmoid_prob, {input_x: x_text_batch, dropout_keep_prob: 1.0,
                                                              embedding_placeholder: embedding_matrix})
                # print(batch_sigmoid_probs)
                if idx == 0:
                    all_predictions = batch_sigmoid_probs
                else:
                    all_predictions = np.concatenate([all_predictions, batch_sigmoid_probs], axis=0)
                # print(batch_sigmoid_probs.shape)
                # print(all_predictions.shape)

    sample_submission = pd.read_csv("../data/sample_submission.csv")
    sample_submission[listed_classes] = all_predictions
    STAMP = "RNN"
    sample_submission.to_csv(STAMP + '.csv', index=False)


def train():
    # Data Preparation
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print("loading data...")
    x, y = load_data_and_labels(FLAGS.data_file, vocab_file)
    print(x.shape, y.shape)
    print("loading embedding")
    embedding_matrix = load_embedding(FLAGS.embd_file)

    # Randomly shuffle data
    np.random.seed(100)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print(x_train.shape, x_dev.shape)
    print(y_train.shape, y_dev.shape)
    print('building models.....')
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = Attention_RNN(
                sequence_length=FLAGS.sequence_length,
                num_classes=y_train.shape[1],
                vocab_size=VOCABULARY_SIZE,
                embedding_size=FLAGS.embedding_dim)

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Ouput directory for models and summaries
            timestamp = str(int(time.time()))
            output_dir = os.path.abspath(os.path.join(os.path.curdir, timestamp))
            print("Writing to {}\n".format(output_dir))

            # Checkpoint directory, Tensorflow assumes this diectory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(output_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                # A single training step
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    model.embedding_placeholder: embedding_matrix
                }

                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, model.loss, model.accuracy],
                    feed_dict)

                return step, loss, accuracy

            def dev_step(x_batch, y_batch):
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.dropout_keep_prob: 1.0,
                    model.embedding_placeholder: embedding_matrix
                }

                step, loss, accuracy, sigmoidprob = sess.run(
                    [global_step, model.loss, model.accuracy, model.sigmoidprob],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                nums = 1
                for l, p in zip(sigmoidprob, y_batch):
                    nums -= 1
                    if nums == 0: break
                    print(l, p)
                print(" Valid Loss {:g}, Valid Acc: {:g}".format(loss, accuracy))

            # training loop, for each batch
            print("start running...")
            for epoch in range(FLAGS.num_epochs):
                batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, 1)
                acum_loss = 0
                acum_ac = 0
                step = 0
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    step += 1
                    if step >=100: break
                    _, loss, accuracy = train_step(x_batch, y_batch)
                    acum_loss += loss
                    acum_ac += accuracy
                    time_str = datetime.datetime.now().isoformat()
                    print("\r {}: step {}, loss {:g}, acc {:g}".format(time_str, step, acum_loss / step,
                                                                       acum_ac / step), end='')
                    sys.stdout.flush()
                    # current_step = tf.train.global_step(sess, global_step)
                print("\r {}, TrainLoss {:g}, TrainAcc {:g}".format(epoch, acum_loss / step, acum_ac / step), end="\t")
                # print("\nEvaluation:")
                dev_step(x_dev, y_dev)
                path = saver.save(sess, checkpoint_prefix, global_step=epoch)
                print("saving to {}".format(path))


if __name__ == '__main__':
    if FLAGS.is_train:
        train()
    else:
        test()

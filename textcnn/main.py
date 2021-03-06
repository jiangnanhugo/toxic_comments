import tensorflow as tf
import numpy as np
import os
import time
from textcnn import TextCNN
from tensorflow.contrib import learn
from builder import *
import datetime
import csv
import sys

tf.flags.DEFINE_boolean("is_train", True, "use the train or test function")

# Parameters
tf.flags.DEFINE_float('dev_sample_percentage', .1, "percentage of the training data to use for validation")
tf.flags.DEFINE_string('data_file', '../input/train.txt', "data source for the file")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default:128)")
tf.flags.DEFINE_integer("sequence_length", 150, "maximum sequence length (default:200)")
tf.flags.DEFINE_string('filter_sizes', '2,3,4,5,7,10,15,20', 'Comma-separated filter sizes (default: "3,4,5")')
tf.flags.DEFINE_integer('num_filters', 10, 'Number of filters per filer size')
tf.flags.DEFINE_float('dropout_keep_prob', 0.1, 'Dropout keep probability (default: 0.5)')
tf.flags.DEFINE_float("l2_reg_lambda", 1e-4, "L2 regularization lambda (default: 0.0)")

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
base_dir = '../input/'
vocab_file = base_dir + "vocabulary.pkl"
embedding_file = base_dir + 'glove.840B.300d.txt'

VOCABULARY_SIZE = 100000 + 1


def test():
    x_test = load_test_data(FLAGS.data_file, vocab_file)
    embedding_matrix = load_embedding(embedding_file, vocab_file)
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
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
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

    sample_submission = pd.read_csv("../input/sample_submission.csv")
    sample_submission[listed_classes] = all_predictions
    STAMP = "textcnn"
    sample_submission.to_csv(STAMP + '.csv', index=False)


def train():
    # Data Preparation
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print("loading data...")
    x, y = load_data_and_labels(FLAGS.data_file, vocab_file)
    print(x.shape, y.shape)
    print("loading embedding")
    embedding_matrix = load_embedding(embedding_file, vocab_file)

    # Randomly shuffle data
    np.random.seed(100)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print('building models.....')
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = TextCNN(
                sequence_length=FLAGS.sequence_length,
                num_classes=y_train.shape[1],
                vocab_size=VOCABULARY_SIZE,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # keep track of gradient values and sparsity (optional)
            print('summaries....')
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)

            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Ouput directory for models and summaries
            timestamp = str(int(time.time()))
            output_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(output_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar('loss', model.loss)
            acc_summary = tf.summary.scalar('accuracy', model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(output_dir, "summaries", 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(output_dir, "summaries", 'dev')
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory, Tensorflow assumes this diectory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(output_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                '''
                A single training step
                '''
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    model.embedding_placeholder: embedding_matrix
                }

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                    feed_dict)

                train_summary_writer.add_summary(summaries, step)
                return step, loss, accuracy

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluate model on a dev set
                """
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.dropout_keep_prob: 1.0,
                    model.embedding_placeholder: embedding_matrix
                }

                step, summaries, loss, accuracy, sigmoidprob = sess.run(
                    [global_step, dev_summary_op, model.loss, model.accuracy, model.sigmoidprob],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                nums = 1
                for l, p in zip(sigmoidprob, y_batch):
                    nums -= 1
                    if nums == 0: break
                    print(l, p)
                print(" Valid Loss {:g}, Valid Acc: {:g}".format(loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

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
                    _, loss, accuracy = train_step(x_batch, y_batch)
                    acum_loss += loss
                    acum_ac += accuracy
                    time_str = datetime.datetime.now().isoformat()
                    print "\r {}: step {}, loss {:g}, acc {:g}".format(time_str, step, acum_loss / step, acum_ac / step),
                    sys.stdout.flush()
                    # current_step = tf.train.global_step(sess, global_step)

                # print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                path = saver.save(sess, checkpoint_prefix, global_step=epoch)
                # print("saved model checkpoint to {}\n".format(path))



if __name__ == '__main__':
    if FLAGS.is_train:
        train()
    else:
        test()

import tensorflow as tf


class Attention_RNN(object):
    def __init__(self, num_classes, embedding_size, vocab_size, sequence_length, num_layers=1):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout')

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.constant(0., shape=[vocab_size, embedding_size]), name="W")  # ,trainable=False)
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size],
                                                        name='embedding_placeholder')
            embeding_init = W.assign(self.embedding_placeholder)
            embedded_inputs = tf.nn.embedding_lookup(embeding_init, self.input_x)

        with tf.name_scope('rnn'):
            # multi rnn
            cells = []
            for _ in range(num_layers):
                cell = tf.contrib.rnn.GRUCell(embedding_size)
                wraped_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
                cells.append(wraped_cell)

            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedded_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]

        # Final (unnormalized scores and predictions)
        with tf.name_scope("output"):
            self.scores = tf.layers.dense(last, num_classes, name='num_classes')
            self.sigmoidprob = tf.sigmoid(self.scores, name='sigmoidprob')
            self.predictions = self.sigmoidprob >= 0.5

        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            #beta=0.795542572957783
            #pos_weight=beta/(1-beta)
            #self.loss_per=tf.nn.weighted_cross_entropy_with_logits(logits=self.scores,targets=self.input_y,pos_weight=pos_weight)
            self.loss_per = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(self.loss_per)#*(1-beta))

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.cast(self.input_y, "bool"))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")

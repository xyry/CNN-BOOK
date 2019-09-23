import tensorflow as tf
# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

class TCNNConfig(object):
    """CNN Configuration parameter"""

    embedding_dim = 64  # Word vector dimension
    seq_length = 200  # Sequence length
    num_classes = 2  # Number of categories

    num_filters = 2048  # Number of convolution kernels

    kernel_size = 5  # Size of convolution kernel

    kernel_size_1 = 2  # kernel size_1
    kernel_size_2 = 3  # kernel size_2
    kernel_size_3 = 4  # kernel size_3
    kernel_size_4 = 5  # kernel size_4

    vocab_size = 5000  # Size of vocabulary

    hidden_dim = 512  # Number of hidden layer neurons
    dropout_keep_prob = 0.5  # dropout keep probability
    learning_rate = 1e-4  #

    batch_size = 64  #
    num_epochs = 10  #

    print_per_batch = 100  #
    save_per_batch = 10  #

    # lambda_loss_amount = 1.28
    # lambda_loss_amount = 0.0001


class TextCNN(object):
    """CNN Model"""

    def __init__(self, config):
        self.config = config


        # self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        # self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        # self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()


    def cnn(self):
        """CNN Model"""
        # Word vector mapping
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim],
                                        initializer=tf.glorot_normal_initializer(seed=1))
            # tf.get_variable(name,  shape, initializer):
            # If initializer is None (the default), the default initializer passed in the variable scope will be used.
            # If that one is None too, a glorot_uniform_initializer will be used.
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            print("embedding_inputs:",embedding_inputs.shape)

        with tf.name_scope("cnn"):

            conv1=tf.layers.conv1d(embedding_inputs,self.config.num_filters/4,self.config.kernel_size_1,name='conv1'
                                   )
                                   # ,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)) #199x64
            conv2=tf.layers.conv1d(embedding_inputs,self.config.num_filters/4,self.config.kernel_size_2,name='conv2'
                                   )
                                   # ,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)) #198x64
            conv3=tf.layers.conv1d(embedding_inputs,self.config.num_filters/4,self.config.kernel_size_3,name='conv3'
                                   )
                                   # ,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)) #197x64
            conv4=tf.layers.conv1d(embedding_inputs,self.config.num_filters/4,self.config.kernel_size_4,name='conv4'
                                   )
                                   # ,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)) #196x64
            print("conv1:",conv1.shape)
            print("conv2:",conv2.shape)
            print("conv3:",conv3.shape)
            print("conv4:",conv4.shape)
            # Perform max-pooling on the column
            gmp1 = tf.reduce_max(conv1, reduction_indices=[1], name='gmp1')  # size=2 max-pooling
            gmp2 = tf.reduce_max(conv2, reduction_indices=[1], name='gmp2')  # size=3 max-pooling
            gmp3 = tf.reduce_max(conv3, reduction_indices=[1], name='gmp3')  # size=4 max-pooling
            gmp4 = tf.reduce_max(conv4, reduction_indices=[1], name='gmp4')  # size=5 max-pooling
            print("gmp1:", gmp1.shape)
            print("gmp2:", gmp2.shape)
            print("gmp3:", gmp3.shape)
            print("gmp4:", gmp4.shape)
            gmp = tf.concat([gmp1, gmp2, gmp3, gmp4], 1)  # concate
            print("gmp:", gmp.shape)




        with tf.name_scope("score"):
            # full connection layer, followed by dropout and relu activation
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            print("fc:",fc.shape)


            # Classifier
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            print("logits:",self.logits.shape)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  #
            self.prob = tf.nn.softmax(self.logits)

        with tf.name_scope("optimize"):
            # Loss ，entropy

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # acc
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

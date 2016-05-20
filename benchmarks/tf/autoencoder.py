#!/usr/bin/env python
"""Trains sparse auto-encoder
"""

import argparse
import math
import numpy as np
import sys
import tensorflow as tf
import time

class FeedForwardNetwork(object):
    """Constructs a basic multi-layer neural network.
    """

    def __init__(self, dim_x, dim_y, hidden_units, layers,
                gpu_mrr=True,
                activation=tf.nn.sigmoid):
        with tf.variable_scope("FFN"):
            # Create input/output variables
            self.x = x = tf.placeholder("float", shape=[None, dim_x])
            self.y_ = y_ = tf.placeholder("float", shape=[None, dim_y])

            # Create model: parameterized for k deep FF layers
            Hsize = [dim_x] + [hidden_units]*layers + [dim_y]
            print "Layers: %s" % str(Hsize)
            k = len(Hsize)-1
            Wall = [None] * k 
            ball = [None] * k
            for (layer, d1) in enumerate(Hsize[:-1]):
                d2 = Hsize[layer+1]
                Wall[layer] = tf.Variable(tf.random_normal(shape=[d1,d2],stddev=0.1))
                ball[layer] = tf.Variable(tf.constant(0.1,shape=[d2]))
            Hact = [None] * (k+1)
            Hact[0] = x
            for layer in range(k):
                Hact[layer+1] = activation(tf.matmul(Hact[layer],Wall[layer]) + ball[layer])
            # output is the last activation
            self.output = y = Hact[k]

            # Loss: numerically stable cross-entropy
            self.loss = loss = -tf.reduce_mean(y_*tf.log(y) + 
                    (tf.sub(1.0,y_)*tf.log(tf.sub(1.000001,y))))

            # Optimizer
            self.lr = tf.Variable(1e-4, trainable=False)
            #self.train_step = tf.train.MomentumOptimizer(self.lr,momentum=0.9).minimize(loss) 
            # Momentum gives very poor results in my experience here.
            #self.train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)
            self.train_step = tf.train.RMSPropOptimizer(self.lr,decay=0.9).minimize(loss)

            self.avgloss = tf.reduce_mean(loss)
 
 

class DataManager(object):
    """Encapsulates low-level data loading.
    """

    def __init__(self, width):
        """Initialize loader
        width: the possible number of bits, which is the dimensionality of the
        vectors
        """
        self.width = width  # number of dimensions
        self.word_assignments = {}  # maps from word to vector index
        self.W = None
 

    def index_for_word(self, word):
        """returns a list of k indices into the output vector
        corresponding to the bits for this word
        """
        if not self.word_assignments.has_key(word):
            idx = len(self.word_assignments)
            self.word_assignments[word] = idx
        return self.word_assignments[word]
 

    def set_bit(self,row,word):
        bit = self.index_for_word(word)
        row[0,bit] = 1
        return row
 

    def parse_line_into_words(self, line):
        """This is specific to the ReMo AIV format, but can be overridden
        """
        line = line.split("\t")[1]  # strip first column, which is customer id
        words = [x[:x.find(",")] for x in line.split(":")]
        return words

    def parse_cust_id(self, line):
        cust_id = line.split("\t")[0]
        return cust_id

    def load(self, filename):
        W_list = []
        with open(filename,"r") as f:
            for line in f.readlines():
                words = self.parse_line_into_words(line)
                row = np.zeros((1,self.width))
                for word in words:
                    row = self.set_bit(row,word)
                W_list.append(row)
        
            self.W = np.concatenate(W_list)
            return self.W
 

class MiniBatcher(object):
    """Iterable set of input/output matrices for training or testing
    """
    def __init__(self, x, y):
        self.batch_pos = 0
        self.x = x
        self.y = y
        self.size = x.shape[0]
        if y.shape[0] != self.size:
            raise RuntimeError("X & Y must have same number of entries")
        
    def next(self, n):
        """Generates the next minibatch of n items.
        Returns a tuple of (x,y)
        """
        if self.batch_pos + n > self.size:
            # We could be cleaner about wrapping
            self.batch_pos = 0
        b = []
        p1 = self.batch_pos
        p2 = p1 + n
        b.append( self.x[p1:p2] )
        b.append( self.y[p1:p2] )
        self.batch_pos = p2
        return b
 

class AutoencoderParser(object):
    """Responsible for loading a directory of data files
    (train/validate,input/output/etc).
    """

    def __init__(self, cmd):
        """Takes a argparse command as configuration.
        Loads data, and makes it accessible as member variables:
        Accessible members:
            train: MiniBatcher object for training
        """
        # Parse config from command
        dims = cmd.vocab_size

        # Set up loader
        mgr = DataManager(dims)

        # Load train data
        train_x = mgr.load(cmd.datafile)
        train_y = train_x
        self.train = MiniBatcher(train_x,train_y)
 

def main(cmd):
    print("Loading datasets")
    all_data = AutoencoderParser(cmd)

    dims = cmd.vocab_size

    print("Constructing neural network")
    dnn = FeedForwardNetwork(dims, dims, cmd.hidden_units, cmd.layers)

    print("Initializing TensorFlow")
    # train the model
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    print("Starting training")
    with sess.as_default():
        start_time = time.time()
        for i in range(cmd.max_iters):
            batch = all_data.train.next(cmd.batch_size)
            train_dict = {
                dnn.x: batch[0], 
                dnn.y_: batch[1],
                dnn.lr: cmd.learning_rate,
            }
            dnn.train_step.run(feed_dict=train_dict)
            if i%cmd.eval_iters == 0:
                spd = (i+1) / (time.time() - start_time)
                print("Iter %d. %giter/s" % (i,spd))
        print "Done training\n"
 
 

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-l","--layers",
            help="Number of hidden layers",
            type=int,
            default=1)
    parser.add_argument("--vocab_size",
            help="Number of unique items to auto-encode",
            type=int,
            default=30000)
    parser.add_argument("-u","--hidden_units",
            help="Size of hidden layer",
            type=int,
            default=8192)
    parser.add_argument("-i","--max_iters",
            help="Maximum number of iterations",
            type=int,
            default=1000)
    parser.add_argument("-b","--batch_size",
            help="minibatch size",
            type=int,
            default=512)
    parser.add_argument("-f","--datafile",
            help="file with input/output data for autoencoder",
            required=True)
    parser.add_argument("-v","--eval_iters",
            help="how often to print speed",
            type=int,
            default=5)
    parser.add_argument("--learning_rate",
            help="learning rate",
            type=float,
            default=1e-4)
    return parser
 

if __name__ == "__main__":
    cmd = get_parser().parse_args()
    main(cmd)

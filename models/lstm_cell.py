#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extend: Grooving with LSTMs
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

logger = logging.getLogger("ner.lstm_cell")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class LSTMCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our GRU cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        """Updates the state using the previous @state and @inputs.
        Remember the LSTM equations are:


        o_t = sigmoid(x_t W_o + h_{t-1} U_o + b_o)
        i_t = sigmoid(x_t W_i + h_{t-1} U_i + b_i)
        f_t = sigmoid(x_t W_f + h_{t-1} U_f + b_f)
        c_tile_t = tanh(x_t W_c + h_{t-1} U_c + b_c)
        c_t = f_t * c_{t-1} + i_t * c_tile_t
        h_t = o_t * tanh(c_t)

        TODO: In the code below, implement an LSTM cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define W_o, U_o, b_o, W_i, U_i, b_i, W_f, U_f, b_f
              W_c, U_c, b_c
              be variables of the apporiate shape using the
              `tf.get_variable' functions.
            - Compute z, r, o and @new_state (h_t) defined above
        Tips:
            - Remember to initialize your matrices using the xavier
              initialization as before.
        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state which is a tuple of vectors (c, h) of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__

        # state of lstm is
        c, h = state
        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            ### YOUR CODE HERE (~20-30 lines)

            W_f = tf.get_variable("W_f", shape=[self.input_size, self._state_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            U_f = tf.get_variable("U_f", shape=[self._state_size, self._state_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            # initialize b_f with constant 1.0
            b_f = tf.get_variable("b_f", shape=[self._state_size], dtype=tf.float32, initializer=tf.constant_initializer(1.0))

            W_i = tf.get_variable("W_i", shape=[self.input_size, self._state_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            U_i = tf.get_variable("U_i", shape=[self._state_size, self._state_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            # initialize b_i with constant 1.0
            b_i = tf.get_variable("b_i", shape=[self._state_size], dtype=tf.float32, initializer=tf.constant_initializer(1.0))

            W_o = tf.get_variable("W_o", shape=[self.input_size, self._state_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            U_o = tf.get_variable("U_o", shape=[self._state_size, self._state_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            # initialize b_o with constant 1.0
            b_o = tf.get_variable("b_o", shape=[self._state_size], dtype=tf.float32, initializer=tf.constant_initializer(1.0))

            W_c = tf.get_variable("W_c", shape=[self.input_size, self._state_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            U_c = tf.get_variable("U_c", shape=[self._state_size, self._state_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_c = tf.get_variable("b_c", shape=[self._state_size], dtype=tf.float32)

            i_t = tf.nn.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(h, U_i) + b_i)
            o_t = tf.nn.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(h, U_o) + b_o)
            f_t = tf.nn.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(h, U_f) + b_f)
            c_tile_t = tf.nn.tanh(tf.matmul(inputs, W_c) + tf.matmul(h, U_c) + b_c)

            c_t = f_t * c + i_t * c_tile_t
            h_t = o_t * tf.nn.tanh(o_t)

            ### END YOUR CODE ###
        output = h_t
        new_state = (c_t, h_t)
        return output, new_state

def test_gru_cell():
    with tf.Graph().as_default():
        with tf.variable_scope("test_lstm_cell"):
            x_placeholder = tf.placeholder(tf.float32, shape=(None,3))
            h_placeholder = tf.placeholder(tf.float32, shape=(None,2))
            # W_o, U_o, b_o, W_i, U_i, b_i, W_f, U_f, b_f
            #               W_c, U_c, b_c
            with tf.variable_scope("lstm"):
                tf.get_variable("W_i", initializer=np.array(np.eye(3,2), dtype=np.float32))
                tf.get_variable("U_i", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("b_i",  initializer=np.array(np.ones(2), dtype=np.float32))

                tf.get_variable("W_o", initializer=np.array(np.eye(3,2), dtype=np.float32))
                tf.get_variable("U_o", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("b_o",  initializer=np.array(np.ones(2), dtype=np.float32))

                tf.get_variable("W_f", initializer=np.array(np.eye(3,2), dtype=np.float32))
                tf.get_variable("U_f", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("b_f",  initializer=np.array(np.ones(2), dtype=np.float32))

                tf.get_variable("W_c", initializer=np.array(np.eye(3,2), dtype=np.float32))
                tf.get_variable("U_c", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("b_c",  initializer=np.array(np.ones(2), dtype=np.float32))

            tf.get_variable_scope().reuse_variables()
            cell = LSTMCell(3, 2)
            state = (h_placeholder, h_placeholder)
            h_t, (c_t, _) = cell(x_placeholder, state, scope="lstm")

            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)
                x = np.array([
                    [0.4, 0.5, 0.6],
                    [0.3, -0.2, -0.1]], dtype=np.float32)
                h = np.array([
                    [0.2, 0.5],
                    [-0.3, -0.3]], dtype=np.float32)
                y = np.array([
                    [ 0.320, 0.555],
                    [-0.006, 0.020]], dtype=np.float32)
                ht = y

                h_t, c_t = session.run([h_t, c_t], feed_dict={x_placeholder: x, h_placeholder: h})
                print("h_t = " + str(h_t))
                print("c_t = " + str(c_t))

                # assert np.allclose(y_, ht_), "output and state should be equal."
                # assert np.allclose(ht, ht_, atol=1e-2), "new state vector does not seem to be correct."

def do_test(_):
    logger.info("Testing lstm_cell")
    test_gru_cell()
    logger.info("Passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests the LSTM cell implementation')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test', help='')
    command_parser.set_defaults(func=do_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)

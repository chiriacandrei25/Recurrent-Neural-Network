# encoding: UTF-8
# Copyright 2017 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
import my_txtutils
import checker

# these must match what was saved !
ALPHASIZE = my_txtutils.ALPHASIZE
directory = "C:/Users/achiriac/Desktop/Workspace/RNN/validation_test/*.txt"
NLAYERS = 3
INTERNALSIZE = 512

eC0 = "C:/Users/achiriac/Desktop/Workspace/RNN/checkpoints/rnn_train_1535104117-12000000"
eC1 = "C:/Users/achiriac/Desktop/Workspace/RNN/checkpoints/rnn_train_1535360733-5800000"

author = eC0
author1 = eC1


def validate_test():
    validate_on_network(author)
    validate_on_network(author1)


def validate_on_network(auth):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(auth + '.meta')
        new_saver.restore(sess, auth)
        valitext, _, __ = my_txtutils.read_data_files(directory, validation=False)

        VALI_SEQLEN = 1 * 64  # Sequence length for validation. State will be wrong at the start of each sequence.
        bsize = len(valitext) // VALI_SEQLEN
        vali_x, vali_y, _ = next(
            my_txtutils.rnn_minibatch_sequencer(valitext, bsize, VALI_SEQLEN, 1))  # all data in 1 batch
        vali_nullstate = np.zeros([bsize, INTERNALSIZE * NLAYERS])
        feed_dict = {'inputs/X:0': vali_x, 'target/Y_:0': vali_y, 'model/pkeep:0': 1.0,
                     'hidden_state/Hin:0': vali_nullstate, 'model/batchsize:0': bsize}

        ls, acc = sess.run(["display_data/batchloss:0", "display_data/accuracy:0"], feed_dict=feed_dict)
        my_txtutils.print_validation_stats(ls, acc)

def generate_text():
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(author + '.meta')
        new_saver.restore(sess, author)
        x = my_txtutils.convert_from_alphabet(ord("E"))
        x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        # initial values
        y = x
        h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
        for i in range(1000000000):
            yo, h = sess.run(['softmax_layer/Yo:0', 'GRU/H:0'],
                             feed_dict={'inputs/X:0': y, 'model/pkeep:0': 1., 'hidden_state/Hin:0': h,
                                        'model/batchsize:0': 1})

            # If sampling is be done from the topn most likely characters, the generated text
            # is more credible. If topn is not set, it defaults to the full distribution (ALPHASIZE)

            # Recommended: topn = 10 for intermediate checkpoints, topn=2 or 3 for fully trained checkpoints

            c = my_txtutils.sample_from_probabilities(yo, topn=2)
            y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
            c = chr(my_txtutils.convert_to_alphabet(c))
            print(c, end="")

validate_test()
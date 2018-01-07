import numpy as np
import tensorflow as tf
from .urnn_cell import URNNCell
import pdb

# add by xiuyi
# add a loss that penalize the unregular angle prediction of RNN
def unreg_loss(pred, alpha=0.3):
    # pred is RNN prediciton, alpha is coeficient balance this loss with other losses.
    out = pred[:, 1:] - pred[:, 0:-1]
    mean, var = tf.nn.moments(out[0, 0:5], axes=1)
    
    return alpha*(tf.reduce_sum(var))    


def serialize_to_file(loss):
    file=open('losses.txt', 'w')
    for l in loss:
        file.write("{0}\n".format(l))
    file.close()

class TFRNN:
    def __init__(
        self,
        name,
        rnn_cell,
        num_in,
        num_hidden, 
        num_out,
        num_target,
        single_output,
        activation_hidden,
        activation_out,
        optimizer,
        loss_function):

        # self
        self.name = name
        self.loss_list = []
        self.init_state_C = np.sqrt(3.0 / (2 * num_hidden))
        self.log_dir = './logs/'
        self.writer = tf.summary.FileWriter(self.log_dir)


        # init cell
        if rnn_cell == URNNCell:
            self.cell = rnn_cell(num_units = num_hidden, num_in = num_in)
        else:
            self.cell = rnn_cell(num_units = num_hidden, activation = activation_hidden)

        # extract output size
        self.output_size = self.cell.output_size
        if type(self.output_size) == dict:
            self.output_size = self.output_size['num_units']
      
        # input_x: [batch_size, max_time, num_in]
        # input_y: [batch_size, max_time, num_target] or [batch_size, num_target]
        self.input_x = tf.placeholder(tf.float32, [None, None, num_in], name="input_x_"+self.name)
        self.input_y = tf.placeholder(tf.float32, [None, num_target] if single_output else [None, None, num_target],  
                                      name="input_y_"+self.name)
        self.keep_prob = tf.placeholder(tf.float32)     

        # add by xiuyi
        # add some conv layer for feature extraction and noise removing

        #pdb.set_trace()
        self.w_conv1 = tf.get_variable("w_conv1"+self.name, shape=(5, num_in, 64), 
                                            initializer=tf.contrib.layers.xavier_initializer()) # fixm
        a1 = tf.nn.conv1d(self.input_x, self.w_conv1, stride=1, padding="SAME")
        a1 = tf.nn.relu(a1)
        a1 = tf.multiply(tf.cast(tf.less_equal(a1, 0.5), tf.float32), a1)

        self.w_conv1_1 = tf.get_variable("w_conv1_1"+self.name, shape=(5, 64, 64), 
                                            initializer=tf.contrib.layers.xavier_initializer()) # fixm
        a1_1 = tf.nn.conv1d(a1, self.w_conv1_1, stride=1, padding="SAME")
        a1_1 = tf.nn.relu(a1_1)
        a1_1 = tf.multiply(tf.cast(tf.less_equal(a1_1, 0.5), tf.float32), a1_1)

        self.w_conv2 = tf.get_variable("w_conv2"+self.name, shape=(11, 64, 64), 
                                            initializer=tf.contrib.layers.xavier_initializer()) # fixm
        a2 = tf.nn.conv1d(a1_1, self.w_conv2, stride=1, padding="SAME")
        a2 = tf.nn.relu(a2)
        a2 = tf.multiply(tf.cast(tf.less_equal(a2, 0.5), tf.float32), a2)
        self.w_conv3 = tf.get_variable("w_conv3"+self.name, shape=(21, 64, 64), 
                                            initializer=tf.contrib.layers.xavier_initializer()) # fixm
        a3 = tf.nn.conv1d(a2, self.w_conv3, stride=2, padding="SAME")
        a3 = tf.nn.relu(a3)
        self.w_conv4 = tf.get_variable("w_conv4"+self.name, shape=(31, 64, 128), 
                                            initializer=tf.contrib.layers.xavier_initializer()) # fixm
        a4 = tf.nn.conv1d(a3, self.w_conv4, stride=5, padding="SAME")
        #pdb.set_trace()
        a4 = tf.nn.relu(a4)

        self.w_conv5 = tf.get_variable("w_conv5"+self.name, shape=(51, 128, 256), 
                                            initializer=tf.contrib.layers.xavier_initializer()) # fixm
        a5 = tf.nn.conv1d(a4, self.w_conv5, stride=5, padding="SAME")
        a5 = tf.nn.relu(a5)

        # rnn initial state(s)
        self.init_states = []
        self.dyn_rnn_init_states = None

        # prepare state size list
        if type(self.cell.state_size) == int:
            state_size_list = [self.cell.state_size]
            self.dyn_rnn_init_states = self.cell.state_size # prepare init state for dyn_rnn
        elif type(self.cell.state_size) == tf.contrib.rnn.LSTMStateTuple:
            state_size_list = list(self.cell.state_size)
            self.dyn_rnn_init_states = self.cell.state_size # prepare init state for dyn_rnn

        # construct placeholder list==
        for state_size in state_size_list:
            init_state = tf.placeholder(tf.float32, [None, state_size], name="init_state")
            self.init_states.append(init_state)
        
        # prepare init state for dyn_rnn
        if type(self.cell.state_size) == int:
            self.dyn_rnn_init_states = self.init_states[0]
        elif type(self.cell.state_size) == tf.contrib.rnn.LSTMStateTuple:
            self.dyn_rnn_init_states = tf.contrib.rnn.LSTMStateTuple(self.init_states[0], self.init_states[1])

        # set up h->o parameters
        self.w_ho = tf.get_variable("w_ho_"+self.name, shape=[num_out, self.output_size], 
                                            initializer=tf.contrib.layers.xavier_initializer()) # fixme
        self.b_o = tf.Variable(tf.zeros([num_out, 1]), name="b_o_"+self.name)

        #import pdb
        #pdb.set_trace()
        # run the dynamic rnn and get hidden layer outputs
        # outputs_h: [batch_size, max_time, self.output_size]
        #outputs_h, final_state = tf.nn.dynamic_rnn(self.cell, self.input_x, initial_state=self.dyn_rnn_init_states) 
        #modified by xiuyi
        outputs_h, final_state = tf.nn.dynamic_rnn(self.cell, a5, initial_state=self.dyn_rnn_init_states)   
       
        # returns (outputs, state)
        #print("after dyn_rnn outputs_h:", outputs_h.shape, outputs_h.dtype)
        #print("after dyn_rnn final_state:", final_state.shape, final_state.dtype)

        # produce final outputs from hidden layer outputs
        if single_output:
            outputs_h = tf.reshape(outputs_h[:, -1, :], [-1, self.output_size])
            # outputs_h: [batch_size, self.output_size]
            preact = tf.matmul(outputs_h, tf.transpose(self.w_ho)) + tf.transpose(self.b_o)
            outputs_o = activation_out(preact) # [batch_size, num_out]
        else:
            # outputs_h: [batch_size, max_time, m_out]
            out_h_mul = tf.einsum('ijk,kl->ijl', outputs_h, tf.transpose(self.w_ho))
            preact = out_h_mul + tf.transpose(self.b_o)
            outputs_o = activation_out(preact) # [batch_size, time_step, num_out]
        self.outputs_o = outputs_o
        # calculate losses and set up optimizer
         
        # add by xiuyi 
        # due emg signal two noise, add a nonlinear and convolution 

        #pdb.set_trace()
        #outputs_o = tf.nn.relu(outputs_o)
        #b, t, o = outputs_o.get_shape().as_list()
        #self.w_sm = tf.get_variable("w_sm_"+self.name, shape=(5, o, o), 
        #                                    initializer=tf.contrib.layers.xavier_initializer()) # fixm
        #outputs_o = tf.nn.conv1d(outputs_o, self.w_sm, stride=1, padding="SAME")

        # loss function is usually one of these two:
        #   tf.nn.sparse_softmax_cross_entropy_with_logits 
        #     (classification, num_out = num_classes, num_target = 1)
        #   tf.squared_difference 
        #     (regression, num_out = num_target)
        #pdb.set_trace()
        if loss_function == tf.squared_difference:
            self.total_loss = tf.add(tf.reduce_mean(loss_function(outputs_o, self.input_y)), unreg_loss(outputs_o))
        elif loss_function == tf.nn.sparse_softmax_cross_entropy_with_logits:
            prepared_labels = tf.cast(tf.squeeze(self.input_y), tf.int32)
            self.total_loss = tf.reduce_mean(loss_function(logits=outputs_o, labels=prepared_labels))
        else:
            raise Exception('New loss function')
        self.train_step = optimizer.minimize(self.total_loss, name='Optimizer')

        # tensorboard
        self.writer.add_graph(tf.get_default_graph())
        self.writer.flush()
        self.writer.close()

        # number of trainable params
        t_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('Network __init__ over. Number of trainable params=', t_params)

    def train(self, dataset, batch_size, epochs):
        # session
        with tf.Session() as sess:
            # initialize global vars
            sess.run(tf.global_variables_initializer())

            # fetch validation and test sets
            num_batches = dataset.get_batch_count(batch_size)
            X_val, Y_val = dataset.get_validation_data()

            # init loss list
            self.loss_list = []
            print("Starting training for", self.name)
            print("NumEpochs:", '{0:3d}'.format(epochs), 
                  "|BatchSize:", '{0:3d}'.format(batch_size), 
                  "|NumBatches:", '{0:5d}'.format(num_batches),'\n')

            # train for several epochs
            for epoch_idx in range(epochs):

                print("Epoch Starting:", epoch_idx, '\n')
                # train on several minibatches
                for batch_idx in range(num_batches):

                    # get one batch of data
                    # X_batch: [batch_size x time x num_in]
                    # Y_batch: [batch_size x time x num_target] or [batch_size x num_target] (single_output?)
                    #pdb.set_trace() 
                    X_batch, Y_batch = dataset.get_batch(batch_idx, batch_size)
                 
                    # evaluate
                    batch_loss = self.evaluate(sess, X_batch, Y_batch, training=True)

                    # save the loss for later
                    self.loss_list.append(batch_loss)

                    # plot
                    if batch_idx%10 == 0:
                        total_examples = batch_size * num_batches * epoch_idx + batch_size * batch_idx + batch_size
                        
                        # print stats
                        serialize_to_file(self.loss_list)
                        print("Epoch:", '{0:3d}'.format(epoch_idx), 
                              "|Batch:", '{0:3d}'.format(batch_idx), 
                              "|TotalExamples:", '{0:5d}'.format(total_examples), # total training examples
                              "|BatchLoss:", '{0:8.4f}'.format(float(batch_loss)))

                # validate after each epoch
                validation_loss = self.evaluate(sess, X_val, Y_val)
                mean_epoch_loss = np.mean(self.loss_list[-num_batches:])
                print("Epoch Over:", '{0:3d}'.format(epoch_idx), 
                      "|MeanEpochLoss:", '{0:8.4f}'.format(mean_epoch_loss),
                      "|ValidationSetLoss:", '{0:8.4f}'.format(validation_loss),'\n')

            self.plot_res(sess, X_val, Y_val)

    def plot_res(self, sess, X, Y):

        # fill (X,Y) placeholders
        feed_dict = {self.input_x: X, self.input_y: Y, self.keep_prob:1.0}
        batch_size = X.shape[0]

        # fill initial state
        for init_state in self.init_states:
            # init_state: [batch_size x cell.state_size[i]]
            feed_dict[init_state] = np.random.uniform(-self.init_state_C, self.init_state_C, [batch_size, init_state.shape[1]])

        outputs_o = sess.run([self.outputs_o], feed_dict)[0]
        #pdb.set_trace()
        import matplotlib.pyplot as plt
        net_o = np.reshape(outputs_o, [-1, 1])
        y = np.reshape(Y, [-1, 1])
        plt.plot(net_o, 'r-')
        plt.plot(y, 'b-')
        plt.show()


    def test(self, dataset, batch_size, epochs):
        # session
        with tf.Session() as sess:
            # initialize global vars
            sess.run(tf.global_variables_initializer())

            # fetch validation and test sets
            X_test, Y_test = dataset.get_test_data()

            test_loss = self.evaluate(sess, X_test, Y_test)
            print("Test set loss:", test_loss)

    def evaluate(self, sess, X, Y, training=False):

        # fill (X,Y) placeholders
        feed_dict = {self.input_x: X, self.input_y: Y}
        batch_size = X.shape[0]
        
        #pdb.set_trace()
        # fill initial state
        for init_state in self.init_states:
            # init_state: [batch_size x cell.state_size[i]]
            feed_dict[init_state] = np.random.uniform(-self.init_state_C, self.init_state_C, [batch_size, init_state.shape[1]])
        #pdb.set_trace()
        # run and return the loss
        if training:
            feed_dict[self.keep_prob]=0.75
            loss, _ = sess.run([self.total_loss, self.train_step], feed_dict)
        else:
            feed_dict[self.keep_prob]=1.0
            loss = sess.run([self.total_loss], feed_dict)[0]
        return loss

    # loss list getter
    def get_loss_list(self):
        return self.loss_list

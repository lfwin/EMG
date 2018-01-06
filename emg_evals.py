import tensorflow as tf
from problems.emg import EMGProblemDataset
from models.tf_rnn import TFRNN
from models.urnn_cell import URNNCell
import numpy as np

loss_path='emg_results/'

glob_learning_rate = 0.001
glob_decay = 0.90

def serialize_loss(loss, name):
    file=open(loss_path + name, 'w')
    for l in loss:
        file.write("{0}\n".format(l))

class Main:
    def init_data(self):
        print('Generating data...')

        self.batch_size=16
        self.epochs=30
        self.data=EMGProblemDataset(-1, -1)

        print('Done.')

    def train_network(self, net, dataset, batch_size, epochs):
        sample_len = str(dataset.get_sample_len())
        print('Training network ', net.name, '... timesteps=',sample_len)
        net.train(dataset, batch_size, epochs)
        # loss_list has one number for each batch (step)
        serialize_loss(net.get_loss_list(), net.name + sample_len)
        print('Training network ', net.name, ' done.')

    def train(self):
        print('Initializing and training URNNs for one timestep...')

        tf.reset_default_graph()
        
        self.emg_lstm=TFRNN(
            name="emg_lstm",
            num_in=1,
            num_hidden=64,
            num_out=1,
            num_target=1,
            single_output=False,
            rnn_cell=tf.contrib.rnn.LSTMCell,
            activation_hidden=tf.tanh,
            activation_out=tf.identity,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
            loss_function=tf.squared_difference)
        self.train_network(self.emg_lstm, self.data, 
                           self.batch_size, self.epochs)
        """        
        tf.reset_default_graph()
        self.emg_urnn=TFRNN(
            name="emg_urnn",
            num_in=5,
            num_hidden=64,
            num_out=2,
            num_target=2,
            single_output=True,
            rnn_cell=URNNCell,
            activation_hidden=None, # modReLU
            activation_out=tf.identity,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
            loss_function=tf.squared_difference)

        self.train_network(self.emg_urnn, self.data, 
                           self.batch_size, self.epochs)
        """

    def train_networks(self):
        print('Starting training...')
        main.train()
        print('Done and done.')

main=Main()
main.init_data()
main.train_networks()

import tensorflow as tf

def build_rnn_graph_lstm(self, inputs, config, is_training):
    """ feedback the output of cell to hidden state
    """
	cell = tf.contrib.rnn.MultiRNNCell(
	    [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

	self._initial_state = cell.zero_state(config.batch_size, data_type())
	state = self._initial_state
	# Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
	# This builds an unrolled LSTM for tutorial purposes only.
	# In general, use the rnn() or state_saving_rnn() from rnn.py.
	#
	# The alternative version of the code below is:
	#
	# inputs = tf.unstack(inputs, num=num_steps, axis=1)
	# outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
	#                            initial_state=self._initial_state)
	outputs = []
	with tf.variable_scope("RNN"):
	  for time_step in range(self.num_steps):
	    if time_step > 0: tf.get_variable_scope().reuse_variables()
	    (cell_output, state) = cell(inputs[:, time_step, :], state)
	    outputs.append(cell_output)
	output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
	
    return output, state

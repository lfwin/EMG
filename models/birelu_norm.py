"""A class for managing bi_relu_norm.                   

This class is designed to ...     
modified by https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow                       

Important use note:  ...                              

"""
import numpy as np
import tensorflow as tf

def get_incoming_shape(incoming):
  """ Returns the incoming data shape """
  if isinstance(incoming, tf.Tensor):
      return incoming.get_shape().as_list()
  elif type(incoming) in [np.array, list, tuple]:
      return np.shape(incoming)
  else:
      raise Exception("Invalid incoming layer.")

# scale and shift not implemented
def normalize(pos, neg, m_pos, m_neg, var_x, scale_after_norm, gamma, epsilon, scope="normalize"):
    input_shape = get_incoming_shape(pos)
    with tf.variable_scope(scope):
      norm = tf.sqrt(tf.add(2*tf.multiply(m_pos, m_neg), var_x))
      
      pos = tf.divide(pos, tf.sqrt(norm+epsilon))
      neg = tf.divide(neg, tf.sqrt(norm+epsilon))

      if scale_after_norm:
          pos = tf.multiply(gamma, pos)
          neg = tf.multiply(gamma, neg)

      res = tf.concat([pos, neg], len(input_shape)-1)
      
      return res      

def bi_relu_norm(x, train, epsilon=1e-7, scale_after_norm=True, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    input_shape = get_incoming_shape(x)
    depth = input_shape[-1]
	#pdb.set_trace()
    axis = range(len(input_shape)-1)
    with tf.variable_scope(scope):
        pos = tf.nn.relu(x)
        neg = tf.negative(tf.nn.relu(tf.negative(x)))

        _, batch_var_x = tf.nn.moments(x, axes=axis) 
        batch_m_pos, batch_var_pos = tf.nn.moments(pos, axes=axis) 
        batch_m_neg, batch_var_neg = tf.nn.moments(neg, axes=axis) 

        #beta = tf.Variable(tf.constant(0.0, shape=[depth]))
        gamma = tf.Variable(tf.constant(1.0, shape=[depth]))
        epsilon = epsilon
        scale_after_norm = scale_after_norm

        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_m_pos, batch_m_neg, batch_var_pos, batch_var_neg, batch_var_x])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_m_pos), tf.identity(batch_m_neg), tf.identity(batch_var_pos), \
                       tf.identity(batch_var_neg), tf.identity(batch_var_x)

        m_pos, m_neg, var_pos, var_neg, var_x = tf.cond(train,
                            mean_var_with_update, lambda: (ema.average(batch_m_pos), \
                                                ema.average(batch_m_neg), ema.average(batch_var_pos), \
                                                ema.average(batch_var_neg), ema.average(batch_var_x)))

        normed = normalize(pos, neg, m_pos, m_neg, var_x, scale_after_norm, gamma, epsilon)

    return normed


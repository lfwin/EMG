import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

def smooth(y, box_pts):
    y_smooth = np.zeros_like(y)
    box = np.ones(box_pts)/box_pts
    if y.shape[1] > 1:
        for i in range(y.shape[1]):
            y_smooth[:, i] = np.convolve(y[:, i], box, mode='same')

    return y_smooth

S_t = tf.constant(0.0)

def assign(x, y):
    x = y
    
    return x

def exp_moving_average(i, X):
    alpha = 0.95
    c = tf.equal(i, tf.constant(0))
    t_f = lambda S_t: assign(S_t, X[0])
    f_f = lambda S_t: assign(S_t, alpha*X[i] + (1-alpha)*S_t); assign(X[i], S_t)

    tf.cond(c, t_f, f_f)

    return X
    

def smooth_loss(output, target):
    batch_size, max_time, output_size = output.shape
    i = tf.constant(0)
    c = tf.less(i, max_time)
    b = lambda i, output: exp_moving_average(i, output)

    exp_mvg_o = tf.while_loop(c, b, [i])

    return tf.reduce_mean(loss_function(exp_mvg_o, target))


def conv_loss(output, target):
    batch_size, max_time, output_size = output.shape
    i = tf.constant(0)
    c = tf.less(i, max_time)
    b = lambda i, output: exp_moving_average(i, output)

    exp_mvg_o = tf.while_loop(c, b, [i])

    return tf.reduce_mean(loss_function(exp_mvg_o, target))

def unreg_loss(pred, alpha):
    # pred is RNN prediciton, alpha is coeficient balance this loss with other losses.
    out = pred[:, 1:] - pred[:, 0:-1]
    mean, var = tf.nn.moments(out, axes=1)
    
    return alpha*(var[0]+var[1])

output = tf.constant(np.array([1, 2, 3, 4]))
target = tf.constant(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]]))

#batch_size, max_time, output_size = output.shape
max_time = output.shape
i = tf.constant(0)
c = tf.less(i, max_time)
#b = lambda i, output: exp_moving_average(i, output)
m, v = unreg_loss(target, 0.2)
#exp_mvg_o = tf.while_loop(c, b, [i])
exp_mvg_o = np.zeros(4)
with tf.Session() as sess:
    tf.global_variables_initializer()
    
    #for i in range(4):
        #exp_mvg_o[i] = exp_moving_average(i, output)
    #print(exp_mvg_o)






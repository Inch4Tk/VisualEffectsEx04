import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#====== random seeds ==================
# makes sure results remain comparable
np.random.seed(0)
tf.set_random_seed(0)

#====== custom modules ==================
import datasets
import uniio

#====== fetch the data ==================
sdf_data = datasets.read_data_sets('training_data')
n_samples = sdf_data.train.num_examples

#====== your part ==================
sess = tf.InteractiveSession()

#============ model definition =============
# define the computational graph

def reduce_multiply(l):
  x = 1
  for i in l:
    x *= i
  return x

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# pooling with conv2d
def pool_2x2(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

# Returns layer to pass on and output shape
def add_conv_layer(input_data, input_shape, filter_list, target_depth, activation=tf.nn.relu):
  complete_filter = [filter_list[0], filter_list[1], input_shape[2], target_depth]
  W_conv = weight_variable(complete_filter)
  b_conv = bias_variable([target_depth])
  layer = activation(conv2d(input_data, W_conv) + b_conv)
  input_shape[2] = target_depth
  return (layer, input_shape)

def add_pool_layer(input_data, input_shape):
  layer = max_pool_2x2(input_data)
  input_shape[0] /= 2
  input_shape[1] /= 2
  return (layer, input_shape)

def add_deconv_layer(input_data, input_shape, filter_list, target_depth,
                     batch_size, activation=tf.nn.relu):
  W_conv = weight_variable([filter_list[0], filter_list[1], target_depth, input_shape[2]])
  b_conv = bias_variable([target_depth])
  deconv_shape = tf.pack([batch_size, input_shape[0], input_shape[1], target_depth])
  layer = activation(tf.nn.conv2d_transpose(input_data, W_conv, output_shape = deconv_shape, strides=[1,1,1,1], padding='SAME') + b_conv)
  out_shape = [input_shape[0], input_shape[1], target_depth]
  return (layer, out_shape)

def add_unpool_layer(input_data, input_shape, batch_size):
  W_conv = weight_variable([2, 2, input_shape[2], input_shape[2]])
  deconv_shape = tf.pack([batch_size, input_shape[0]*2, input_shape[1]*2, input_shape[2]])
  layer = tf.nn.conv2d_transpose(input_data, W_conv, output_shape = deconv_shape, strides=[1,2,2,1], padding='SAME')
  out_shape = [input_shape[0]*2, input_shape[1]*2, input_shape[2]]
  return (layer, out_shape)

def add_fully_connected(input_data, input_size, target_size, activation=tf.nn.relu):
  W = weight_variable([input_size, target_size])
  b = bias_variable([target_size])
  layer = activation(tf.matmul(input_data, W,) + b)
  return layer

x_image = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
x = tf.reshape(x_image, [-1,12288])
alphas = tf.placeholder(tf.float32, shape=[None, 1])

# Weights and Biases
n_code = 100
batch_size = tf.shape(x_image)[0]

# Dropout
keep_prob = tf.placeholder("float")
x_norm = tf.nn.dropout(x, keep_prob)
x_conv = tf.reshape(x_norm, [-1, 64, 64, 3])

# Build autoencoder
lay1 = add_fully_connected(tf.reshape(x_conv, [-1, 12288]), 12288, 100, tf.nn.tanh)
dlay1 = add_fully_connected(lay1, 100, 12288, tf.nn.tanh)
#lay1, lay1size = add_conv_layer(x_conv, [64, 64, 3], [5, 5], 64)
#lay2, lay2size = add_pool_layer(lay1, lay1size)
#lay3, lay3size = add_conv_layer(lay2, lay2size, [5, 5], 32)
#lay4, lay4size = add_pool_layer(lay3, lay3size)
#lay5 = add_fully_connected(tf.reshape(lay4, [-1, 8192]), 8192, 20, tf.nn.tanh)
#lay6 = add_fully_connected(lay5, 400, 100, tf.nn.relu)
#lay7 = add_fully_connected(lay6, 100, 20, tf.nn.tanh)

#dlay7 = add_fully_connected(lay7, 20, 100, tf.nn.tanh)
#dlay6 = add_fully_connected(dlay7, 100, 400, tf.nn.relu)
#dlay5 = add_fully_connected(lay5, 20, 8192, tf.nn.tanh)
#reshapesize = [-1]
#reshapesize.extend(lay4size)
#dlay4, dlay4size = add_unpool_layer(tf.reshape(dlay5, reshapesize), lay4size, batch_size)
#dlay3, dlay3size = add_deconv_layer(dlay4, dlay4size, [5, 5], 64, batch_size)
#dlay2, dlay2size = add_unpool_layer(dlay3, dlay3size, batch_size)
#dlay1, dlay1size = add_deconv_layer(dlay2, dlay2size, [5, 5], 3, batch_size)

#y_image = dlay1
#y = tf.reshape(dlay1, [-1, 12288])
y_image = tf.reshape(dlay1, [-1, 64, 64, 3])
y = dlay1
#============ training your model =============

l2_loss = tf.nn.l2_loss(y - x)
norm = tf.nn.l2_loss(x)
weight_penalty = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
loss = l2_loss + 0.005*weight_penalty

learning_rate = 1e-4
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

sess.run(init_op)

# train the model
#'''
for i in range(25000):
    batch = sdf_data.train.next_batch(1)
    if i%100 == 0:
        train_loss = loss.eval(feed_dict={x_image:batch[0], alphas: batch[1], keep_prob: 1.0})
        real_loss = l2_loss.eval(feed_dict={x_image:batch[0], alphas: batch[1], keep_prob: 1.0})
        print("step %d, training loss %g, real loss %g"%(i, train_loss, real_loss))
    #if i%5000 == 0 and i > 0:
    #    inp = raw_input("Continue? (anything except n will continue): ")
    #    if inp == "n":
    #        break
    train_step.run(feed_dict={x_image: batch[0], alphas: batch[1], keep_prob: 0.95})

# save the trained model
#model_file = saver.save(sess, "model.ckpt")
#print("Trained model saved to %s"%model_file)
#'''

# alternatively restore the model
# this will be used for your presentation instead of training
#saver.restore(sess, "model.ckpt")

#============ score =============
#Do not alter this part

err = l2_loss.eval(feed_dict={x_image: sdf_data.test.inputs, alphas: sdf_data.test.labels, keep_prob: 1.0})
print("validation loss: %g"%err)

err = err / norm.eval(feed_dict={x_image: sdf_data.test.inputs, alphas: sdf_data.test.labels, keep_prob: 1.0})
score = (1 - n_code / float(64*64*3)) * (1 - err)
print("Your score is %g"%score)

#============ validating your model =============
for i in range(5):
    batch = sdf_data.train.next_batch(1)
    ref = batch[0][0]
    gen = sess.run(y_image, feed_dict={x_image:[ref], keep_prob: 1.0})
       
    fig, [ax1, ax2]= plt.subplots(1, 2, figsize=(6, 3))
    _ = ax1.quiver(ref[:,:,0], ref[:,:,1], pivot='tail', color='k', scale=1 / 1)
    _ = ax2.quiver(gen[0,:,:,0], gen[0,:,:,1], pivot='tail', color='k', scale=1 / 1)

    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, 60)
    ax2.set_xlim(0, 60)
    ax2.set_ylim(0, 60)

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    fig.savefig("derp_%g.png" % i)
    
    train_loss = l2_loss.eval(feed_dict={x_image:[ref], alphas: batch[1], keep_prob: 1.0})
    print("loss on first batches %i, loss %g"%(i, train_loss))
    
for i in range(5):
    ref = sdf_data.test.inputs[i]
    gen = sess.run(y_image, feed_dict={x_image:[ref], keep_prob: 1.0})

    fig, [ax1, ax2]= plt.subplots(1, 2, figsize=(6, 3))
    _ = ax1.quiver(ref[:,:,0], ref[:,:,1], pivot='tail', color='k', scale=1 / 1)
    _ = ax2.quiver(gen[0,:,:,0], gen[0,:,:,1], pivot='tail', color='k', scale=1 / 1)

    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, 60)
    ax2.set_xlim(0, 60)
    ax2.set_ylim(0, 60)

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    plt.show()
		# Docker users should uncomment the following line
		# It will save the graphs to the disk for viewing
    fig.savefig("validate_%g.png" % i)
    train_loss = l2_loss.eval(feed_dict={x_image:[ref], keep_prob: 1.0})
    print("loss on first testdata %i, loss %g"%(i, train_loss))

#====== write the data ==================

print("Saving outputs...")

save_dir = 'output_data'
head, _ = uniio.readuni('./training_data/vel_000000.uni')

all_data = np.concatenate((sdf_data.train.inputs, sdf_data.test.inputs), axis=0)
all_labels = np.concatenate((sdf_data.train.labels, sdf_data.test.labels), axis=0)

N = all_labels.shape[0]

for i in range(N):
    enc = 100*sess.run(y_image, feed_dict={x_image:[all_data[i]], keep_prob: 1.0})
    loc = save_dir + '/vel_%06d.uni' % all_labels[i]
    uniio.writeuni(loc, head, enc)

print("Output data succesfully saved to %s"%save_dir)

sess.close()

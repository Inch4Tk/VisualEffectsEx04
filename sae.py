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

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def deconv2d(x, W, shape):
    return tf.nn.conv2d_transpose(x, W, shape, strides=[1, 1, 1, 1], padding='SAME') 

n_code = 1000

x_image = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
x = tf.reshape(x_image, [-1,12288])
alphas = tf.placeholder(tf.float32, shape=[None, 1])

W_conv1 = weight_variable([10,10,3,3])
b_conv1 = bias_variable([3])
h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1)

W_conv2 = weight_variable([5,5,3,3])
b_conv2 = bias_variable([3])
h_conv2 = tf.nn.tanh(conv2d(h_conv1, W_conv2) + b_conv2)

W_conv3 = weight_variable([3,3,3,3])
b_conv3 = bias_variable([3])
h_conv3 = tf.nn.tanh(conv2d(h_conv2, W_conv3) + b_conv3)

x_1 = tf.reshape(h_conv3,[-1,64*64*3])

W_1 = weight_variable([64*64*3, n_code])
b_1 = bias_variable([n_code])

z = tf.nn.tanh(tf.matmul(x_1, W_1) + b_1)

W_2 = weight_variable([n_code, 64*64*3])
b_2 = bias_variable([64*64*3])
y_1 = tf.nn.tanh(tf.matmul(z,W_2)+b_2)

y_2 = tf.reshape(y_1, [-1,64,64,3])

W_deconv1 = weight_variable([3,3,3,3])
b_deconv1 = bias_variable([3])
h_deconv1 = tf.nn.tanh(deconv2d(y_2,W_deconv1, [1,64,64,3]) + b_deconv1)

W_deconv2 = weight_variable([3,3,3,3])
b_deconv2 = bias_variable([3])
h_deconv2 = tf.nn.tanh(deconv2d(h_deconv1, W_conv2, [1,64,64,3]) + b_conv2)

W_deconv3 = weight_variable([3,3,3,3])
b_deconv3 = bias_variable([3])
h_deconv3 = tf.nn.tanh(deconv2d(h_deconv2, W_conv3, [1,64,64,3]) + b_conv3)

y_image = h_deconv3 #tf.reshape(h_deconv3, [-1,64,64,3])
y = tf.reshape(y_image, [-1,12288])

#============ training your model =============

l2_loss = tf.nn.l2_loss(y - x)
norm = tf.nn.l2_loss(x)
weight_penalty = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
loss = l2_loss + 0.001*weight_penalty

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

sess.run(init_op)

# train the model
#'''
for i in range(50000):
    batch = sdf_data.train.next_batch(1)
    if i%100 == 0:
        train_loss = loss.eval(feed_dict={x_image:batch[0], alphas: batch[1]})
        print("step %d, training loss %g"%(i, train_loss))
    train_step.run(feed_dict={x_image: batch[0], alphas: batch[1]})

# save the trained model
model_file = saver.save(sess, "model.ckpt")
print("Trained model saved to %s"%model_file)
#'''

# alternatively restore the model
# this will be used for your presentation instead of training
#saver.restore(sess, "model.ckpt")

#============ score =============
#Do not alter this part

err = l2_loss.eval(feed_dict={x_image: sdf_data.test.inputs, alphas: sdf_data.test.labels})
print("validation loss: %g"%err)

err = err / norm.eval(feed_dict={x_image: sdf_data.test.inputs, alphas: sdf_data.test.labels})
score = (1 - n_code / float(64*64*3)) * (1 - err)
print("Your score is %g"%score)

#============ validating your model =============

for i in range(5):
    ref = sdf_data.test.inputs[i]
    gen = sess.run(y_image, feed_dict={x_image:[ref]})

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

#====== write the data ==================

print("Saving outputs...")

save_dir = 'output_data'
head, _ = uniio.readuni('./training_data/vel_000000.uni')

all_data = np.concatenate((sdf_data.train.inputs, sdf_data.test.inputs), axis=0)
all_labels = np.concatenate((sdf_data.train.labels, sdf_data.test.labels), axis=0)

N = all_labels.shape[0]

for i in range(N):
    enc = 100*sess.run(y_image, feed_dict={x_image:[all_data[i]]})
    loc = save_dir + '/vel_%06d.uni' % all_labels[i]
    uniio.writeuni(loc, head, enc)

print("Output data succesfully saved to %s"%save_dir)

sess.close()

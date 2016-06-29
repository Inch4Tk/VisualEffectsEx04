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

x_image = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
x = tf.reshape(x_image, [-1,12288])
alphas = tf.placeholder(tf.float32, shape=[None, 1])

# Weights and Biases
n_code = 100
patchsize = 5
num_channels = 3
depth = 32
layer2size = 50
batch_size = tf.shape(x_image)[0]

# Dropout
keep_prob = tf.placeholder("float")
x_norm = tf.nn.dropout(x, keep_prob)
x_conv = tf.reshape(x_norm, [-1, 64, 64, 3])

#======================
# Encoding
# convolution and normal weights
We_1 = weight_variable([patchsize, patchsize, num_channels, depth]) # 3,32
be_1 = bias_variable([depth])

We_2 = weight_variable([patchsize, patchsize, depth, depth]) # 32,32
be_2 = bias_variable([depth])

We_3 = weight_variable([patchsize, patchsize, depth, depth * 2]) # 32, 64
be_3 = bias_variable([depth*2])

We_4 = weight_variable([2, 2, depth*2, depth / 4]) # 64, 8
be_4 = bias_variable([depth / 4])

#We_3 = weight_variable([16*16*depth, 512])
#be_3 = bias_variable([512])

#We_4 = weight_variable([512, 100])
#be_4 = bias_variable([100])

# calc 1st convolution
conv1 = conv2d(x_conv, We_1)
lay1 = tf.nn.relu(conv1 + be_1)
lay1 = max_pool_2x2(lay1)

# calc 2nd convolution
conv2 = conv2d(lay1, We_2)
lay2 = tf.nn.relu(conv2 + be_2)
lay2 = max_pool_2x2(lay2)

# calc 3rd convolution
conv3 = conv2d(lay2, We_3)
lay3 = tf.nn.relu(conv3 + be_3)
lay3 = max_pool_2x2(lay3)

# calc 4th convolution
conv4 = conv2d(lay3, We_4)
lay4 = tf.nn.relu(conv4 + be_4) # encoded layer

# Lowest layer is now: 8x8x8

# reshape results
#lay2_s = lay2.get_shape().as_list()
#lay2_reshape = tf.reshape(lay2, [-1, lay2_s[1] * lay2_s[2] * lay2_s[3]])

# Calculate 2x normal fully connected layers
#lay3 = tf.nn.relu(tf.matmul(lay2_reshape, We_3) + be_3)
#encoded = tf.nn.relu(tf.matmul(lay3, We_4) + be_4)

#======================
# Decoding
# Variables for fully connected layers
#Wd_1 = weight_variable([100, 512])
#bd_1 = bias_variable([512])

#Wd_2 = weight_variable([512, 4096])
#bd_2 = bias_variable([4096])

# Variables for Deconvolutions
Wd_1 = weight_variable([2, 2, depth * 2, depth/4])
bd_1 = bias_variable([depth * 2])
deconv1_shape = tf.pack([batch_size, 8, 8, depth * 2])

# Pool inversion
Wd_2_pool = weight_variable([2, 2, depth * 2, depth * 2])
deconv2_shape_pool = tf.pack([batch_size, 16, 16, depth * 2])
Wd_2 = weight_variable([5, 5, depth, depth * 2])
bd_2 = bias_variable([depth])
deconv2_shape = tf.pack([batch_size, 16, 16, depth])

# Pool inversion
Wd_3_pool = weight_variable([2, 2, depth, depth])
deconv3_shape_pool = tf.pack([batch_size, 32, 32, depth])
Wd_3 = weight_variable([5, 5, depth, depth])
bd_3 = bias_variable([depth])
deconv3_shape = tf.pack([batch_size, 32, 32, depth])

# Pool inversion
Wd_4_pool = weight_variable([2, 2, depth, depth])
deconv4_shape_pool = tf.pack([batch_size, 64, 64, depth])
Wd_4 = weight_variable([5, 5, num_channels, depth])
bd_4 = bias_variable([num_channels])
deconv4_shape = tf.pack([batch_size, 64, 64, num_channels])


# Applying decoding of fully connected layers        
#lay4 = tf.nn.relu(tf.matmul(encoded, Wd_1) + bd_1)
#lay5 = tf.nn.relu(tf.matmul(lay4, Wd_2) + bd_2)

# Reshaping 
#lay5_reshape = tf.reshape(lay5, [-1, 16, 16, 16])

# Applying deconvolutions
# decode layer1
dlay1 = tf.nn.conv2d_transpose(lay4, Wd_1,
                                 output_shape = deconv1_shape,
                                 strides=[1,1,1,1], padding='SAME')
dlay1 = tf.nn.relu(dlay1 + bd_1)

# decode layer1
dlay2_pool = tf.nn.conv2d_transpose(dlay1, Wd_2_pool,
                                 output_shape = deconv2_shape_pool,
                                 strides=[1,2,2,1], padding='SAME')
dlay2 = tf.nn.conv2d_transpose(dlay2_pool, Wd_2,
                                 output_shape = deconv2_shape,
                                 strides=[1,1,1,1], padding='SAME')
dlay2 = tf.nn.relu(dlay2 + bd_2)

# decode layer3
dlay3_pool = tf.nn.conv2d_transpose(dlay2, Wd_3_pool,
                                 output_shape = deconv3_shape_pool,
                                 strides=[1,2,2,1], padding='SAME')
dlay3 = tf.nn.conv2d_transpose(dlay3_pool, Wd_3,
                                 output_shape = deconv3_shape,
                                 strides=[1,1,1,1], padding='SAME')
dlay3 = tf.nn.relu(dlay3 + bd_3)

# decode layer4
dlay4_pool = tf.nn.conv2d_transpose(dlay3, Wd_4_pool,
                                 output_shape = deconv4_shape_pool,
                                 strides=[1,2,2,1], padding='SAME')
dlay4 = tf.nn.conv2d_transpose(dlay4_pool, Wd_4,
                                 output_shape = deconv4_shape,
                                 strides=[1,1,1,1], padding='SAME')
dlay4 = tf.nn.relu(dlay4 + bd_4)


y = tf.reshape(dlay4, [-1,12288])
y_image = dlay4

#z=tf.nn.tanh(tf.matmul(x, W_1) + b_1)
#W_11 = weight_variable([n_code, 150])
#b_11 = bias_variable([150])
#zz = tf.nn.relu(tf.matmul(z, W_11) + b_11)
#W_22 = weight_variable([150, n_code])
#b_22 = bias_variable([n_code])
#yy = tf.nn.relu(tf.matmul(zz, W_22) + b_22)
#W_2 = weight_variable([n_code, 12288])
#b_2 = bias_variable([12288])
#y = tf.nn.tanh(tf.matmul(yy, W_2) + b_2)
#y_image = tf.reshape(y, [-1,64,64,3])

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
for i in range(1000):
    batch = sdf_data.train.next_batch(1)
    if i%100 == 0:
        train_loss = loss.eval(feed_dict={x_image:batch[0], alphas: batch[1], keep_prob: 1.0})
        print("step %d, training loss %g"%(i, train_loss))
    train_step.run(feed_dict={x_image: batch[0], alphas: batch[1], keep_prob: 0.8})

# save the trained model
model_file = saver.save(sess, "model.ckpt")
print("Trained model saved to %s"%model_file)
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

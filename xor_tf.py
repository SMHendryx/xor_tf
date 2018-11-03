# Solves the classic xor problem in tensorflow with a simple feedforward neural network.

# Author: Sean M. Hendryx

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# define model:
class Model:
    def __init__(self):
        self.num_inputs = 2 # dimensionality of the input
        self.num_hidden_units = 5
        self.num_outputs = 1 # dimensionality of the output
        self.W1 = self.init_weights([self.num_inputs, self.num_hidden_units])
        self.b1 = self.init_weights([self.num_hidden_units])
        self.w2 = self.init_weights([self.num_hidden_units, self.num_outputs]) # 2 hidden layers
        self.b2 = self.init_weights([self.num_outputs])

        # Make placeholder tensors:
        self.input_ph = tf.placeholder(tf.float32, shape=(None, self.num_inputs))
        self.label_ph = tf.placeholder(tf.float32, shape=(None, self.num_outputs))

        # Define forward pass:
        self.logits = self.forward(self.input_ph)
        self.learning_rate = 0.1
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_ph, logits=self.logits))
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.prediction = self.forward(self.input_ph, train=False)


    def init_weights(self, input_shape):
        # Initialize variables:
        return tf.Variable(tf.truncated_normal(input_shape))

    def forward(self, X, train=True):
        """
        Simple feedforward neural network.
        :param train: Whether or not to train return logits (for training) or predictions.
        """
        Z1 = tf.nn.sigmoid(tf.matmul(X, self.W1) + self.b1)
        Z2 = tf.matmul(Z1, self.w2) + self.b2
        if train:
            return Z2
        else:
            return tf.nn.sigmoid(Z2)


model = Model()

# Make training data:
X = np.array([[0,0],[1,1],[1,0],[0,1]])
y = np.array([[0], [0], [1], [1]])

num_epochs = 100
eval_interval = 10
costs = []

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    ## Run the Op that initializes global variables.
    sess.run(init_op)
    # Train:
    print('Training...')
    for epoch in range(num_epochs):
        sess.run(model.train, feed_dict={model.input_ph: X, model.label_ph: y})
        loss = sess.run(model.loss, feed_dict={model.input_ph: X, model.label_ph: y})
        costs.append(loss)

        if epoch % eval_interval == 0:
            print("Cost at epoch {}: {}".format(epoch, loss))

    print('Training complete.')

    # Make prediction:
    prediction = sess.run(model.prediction, feed_dict={model.input_ph: X})


print('Labels:')
print(y)
print('Predictions:')
print(prediction)

# Plot training loss:
plt.plot(costs)
plt.show()




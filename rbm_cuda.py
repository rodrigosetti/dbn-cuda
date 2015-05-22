#coding: utf-8

from __future__ import division
import time
import numpy as np
import cudamat as cm
import pyprind

class RestrictedBoltzmanMachine(object):

    def __init__(self, n_hidden, learning_rate=0.1, momentum=0.9, n_epochs=30, batch_size=128, k=1, title=''):
        self.n_hidden      = n_hidden
        self.learning_rate = learning_rate
        self.momentum      = momentum
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.k             = k
        self.title         = title

    def transform(self, v, h):
        """
        Parameters:
        v : the visible input activation
        h : the target to write the hidden activation
        """
        cm.dot(self.W.T, v, target = h)
        h.add_col_vec(self.hidden_bias)
        h.apply_sigmoid()

    def sample_hidden(self, v, h_mean, h):
        """
        Parameters:
        v : the visible input activation
        h_mean : the target to write the hidden activation
        h: the target to write the hidden sample
        """
        self.transform(v, h_mean)
        h.fill_with_rand()
        h.less_than(h_mean)

    def sample_visible(self, h, v_mean, v):
        """
        Parameters:
        h : the hidden activation
        v_mean : the target to write the visible activation
        v: the target to write the visible sample
        """
        self.reverse_transform(h, v_mean)
        v.fill_with_rand()
        v.less_than(v_mean)

    def reverse_transform(self, h, v):
        """
        Parameters:
        h : the hidden activation
        v : the target to write the visible activation
        """
        cm.dot(self.W, h, target = v)
        v.add_col_vec(self.visible_bias)
        v.apply_sigmoid()

    def fit(self, input, verbose=1):
        """
        Parameters
        ----------
        input : CUDAMatrix array, shape (n_components, n_samples) - opposite of scikit-learn
        """
        n_samples = input.shape[1]
        num_batches = n_samples // self.batch_size

        # model parameters
        self.n_visible = input.shape[0]

        # initialize weights
        self.W            = cm.CUDAMatrix(0.1 * np.random.randn(self.n_visible, self.n_hidden))
        self.visible_bias = cm.CUDAMatrix(np.zeros((self.n_visible, 1)))
        self.hidden_bias  = cm.CUDAMatrix(-4.*np.ones((self.n_hidden, 1)))

        # initialize weight updates
        u_W            = cm.CUDAMatrix(np.zeros((self.n_visible , self.n_hidden  )))
        u_visible_bias = cm.CUDAMatrix(np.zeros((self.n_visible , 1)))
        u_hidden_bias  = cm.CUDAMatrix(np.zeros((self.n_hidden  , 1)))

        # initialize temporary storage
        v = cm.empty((self.n_visible, self.batch_size))
        h = cm.empty((self.n_hidden , self.batch_size))
        r = cm.empty((self.n_hidden , self.batch_size))

        if verbose == 1:
            bar = pyprind.ProgBar(self.n_epochs, title=self.title)

        for epoch in range(self.n_epochs):
            start_time = time.time()
            err = []

            for batch in range(num_batches):
                # get current minibatch
                v_true = input.slice(batch*self.batch_size, (batch + 1)*self.batch_size)
                v.assign(v_true)

                # apply momentum
                u_W.mult(self.momentum)
                u_visible_bias.mult(self.momentum)
                u_hidden_bias.mult(self.momentum)

                # positive phase
                self.transform(v, h)

                u_W.add_dot(v, h.T)
                u_visible_bias.add_sums(v, axis = 1)
                u_hidden_bias.add_sums(h, axis = 1)

                # sample hiddens
                r.fill_with_rand()
                r.less_than(h, target = h)

                # negative phase CD-k
                for n in xrange(self.k):
                    self.reverse_transform(h, v)
                    self.transform(v, h)

                u_W.subtract_dot(v, h.T)
                u_visible_bias.add_sums(v , axis = 1, mult = -1.)
                u_hidden_bias.add_sums(h  , axis = 1, mult = -1.)

                # update weights
                self.W.add_mult(u_W, self.learning_rate/self.batch_size)
                self.visible_bias.add_mult(u_visible_bias , self.learning_rate/self.batch_size)
                self.hidden_bias.add_mult(u_hidden_bias   , self.learning_rate/self.batch_size)

                # calculate reconstruction error
                v.subtract(v_true)
                err.append(v.euclid_norm()**2 / (self.n_visible * self.batch_size))

            if verbose == 1:
                bar.update()
            elif verbose > 1:
                print("Epoch: %i, MSE: %.6f, Time: %.6f s" % (epoch+1, np.mean(err), (time.time() - start_time)))

        # frees memory
        u_W.free_device_memory()
        u_visible_bias.free_device_memory()
        u_hidden_bias.free_device_memory()
        v.free_device_memory()
        h.free_device_memory()
        r.free_device_memory()

class DeepBeliefNetwork(object):

    def __init__(self, layers):
        self.layers = layers

    def fit(self, input):
        """
        Train each layer of the network

        Parameters
        ----------
        input: A CUDAMatrix shaped as (n_features, n_samples)
        """
        n_samples = input.shape[1]

        for n, layer in enumerate(self.layers):
            layer.fit(input)

            if n+1 < len(self.layers):
                h = cm.empty((layer.n_hidden, n_samples))
                layer.transform(input, h)

                if n > 0:
                    input.free_device_memory()

                input = h

        if len(self.layers) > 1:
            input.free_device_memory()

    def transform(self, input):
        """
        Transform the input through each layer
        Parameters
        ----------
        input: A CUDAMatrix shaped as the first layer

        Return
        ------
        A newly allocated CUDAMatrix with the shape of the last layer.
        """
        n_samples = input.shape[1]
        for n, layer in enumerate(layers):
            h = cm.empty((layer.n_hidden, n_samples))
            layer.transform(input, h)

            if n > 0:
                input.free_device_memory()

            input = h

        return input

    def reverse_transform(self, h):
        """
        Reverse transform from last to first layer

        Parameters
        ----------
        h: A CUDAMatrix shaped as the last layer

        Return
        ------
        A new CUDAMatrix with the shape of the first layer
        """
        for n, layer in enumerate(reversed(self.layers)):
            v = cm.empty(layer.visible_bias.shape)
            layer.reverse_transform(h, v)
            if n > 0:
                h.free_device_memory()
            h = v

        return v

    def dream(self, k=10):
        """
        Generate a pattern from this network.
        Return
        ------
        A new CUDAMatrix with the shape of the first layer
        """
        last_layer = self.layers[-1]

        v      = cm.empty(last_layer.visible_bias.shape)
        h      = cm.empty(last_layer.hidden_bias.shape)

        v_mean = cm.empty(last_layer.visible_bias.shape)
        h_mean = cm.empty(last_layer.hidden_bias.shape)

        h.fill_with_rand()
        for _ in xrange(k):
            last_layer.sample_visible(h, v_mean, v)
            last_layer.sample_hidden(v, h_mean, h)

        v.free_device_memory()
        v_mean.free_device_memory()
        h_mean.free_device_memory()

        return self.reverse_transform(h)


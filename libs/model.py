from lib2to3.pgen2.pgen import generate_grammar
import os
import sys
from datetime import datetime
import pickle
import abc
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
tfd = tfp.distributions
import math
from libs.util import print_log, save_weights, load_weights
# from distribution_shift_generator import LongTransformedCifar10Generator
# from distribution_shift_generator import LongTransformedSvhnGenerator
from libs.util import ind_multivariate_normal_fn

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class NoLongerAnImageClassificationModel(abc.ABC):
    """Tensorflow model class with a prior over weights.

    The model is compatible with point estimation (MAP) and Bayesian 
    inference (stochastic variational inference). Both paramdigms share the
    same loss function structure, i.e., log_posterior = log_likelihood +
    log_prior. Thus they can use the same code architecture.

    The optimization algorithm is ADAM.
    """
    def __init__(self, x_train, y_train, neural_net, 
                 learning_rate=0.00001, beta = 1., gamma = 1., rng=None):
        self.x_train = x_train
        self.y_train = y_train
        self.beta = beta
        self.gamma = gamma
        assert self.x_train.shape[0] == self.y_train.shape[0]
        self.training_size = self.x_train.shape[0]

        self.learning_rate = learning_rate
        if rng is None:
            self.rng = np.random.RandomState(2**31)
        else:
            self.rng = rng

        with tf.compat.v1.variable_scope('placeholders'):
            self.data = tf.placeholder(
                tf.float32, 
                shape=[None,
                       self.x_train.shape[1]])
            self.labels = tf.placeholder(tf.float32, shape=[None,])
            self.training = tf.placeholder_with_default(False, shape=())

        self.neural_net = neural_net

        self.logits = self.neural_net(self.data, training=self.training)  # mc_num=1

        # training
        with tf.compat.v1.variable_scope('training_loss'):
            self.neg_log_likelihood = tf.math.reduce_mean(-math.log(1/ (math.sqrt(2 * math.pi)* 0.01)) + (1/(0.01**2)) * 0.5 * ((self.labels -self.logits)**2))
            # need to divide the total number of training samples to get the
            # correct scale of the ELBO: 
            # notice that we use `reduce_mean` as above
            self.kl = sum(self.neural_net.losses) / self.training_size
            self.nelbo = self.neg_log_likelihood + self.kl
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=0.00001)
        self.train_step = self.optimizer.minimize(self.nelbo)

        # testing
        self.predictions = self.logits

    # initialization
    def init_session(self):
        # Initializing the variables
        init = tf.global_variables_initializer()

        # launch a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.sess.run(init)

    def test_error(self, x_test, y_test, mc_num=200):
        """The same test data will be evaluated `mc_num` times.

        Args:
        x_test -- numpy array test images of the same shape as x_train
        y_test -- numpy array test image labels of the same shape as y_train
        mc_num -- number of times for which the test dataset will be evaluated.

        Returns:
        Average test accuracy of all images;
        Probabilities of each class for each test image;
        Cross entropy of the test dataset.
        """
        for i in range(mc_num):
            (nlog_lik, y_pred) = self.sess.run(
                [self.neg_log_likelihood, self.logits], 
                feed_dict={
                            self.data: x_test, 
                            self.labels: y_test,
                            self.training: False
                })
        y_pred = np.reshape(y_pred, newshape = y_test.shape)
        #print("y_test_shape", y_test.shape)
        #print("y_pred_Shape", y_pred.shape)
        #print("list_check", abs(y_pred - y_test)) 
        #print("list_checks_shape", abs(y_pred-y_test).shape)
        #print("list_final", np.mean(abs(y_pred - y_test)))   
        test_abserr = np.mean(abs(y_pred - y_test))
        return test_abserr, y_pred

    def get_elbo(self, batch_size=64, mc_num=10000):
        elbo = 0.
        for i in range(mc_num):
            random_batch_ind = self.rng.randint(
                low=0, 
                high=self.training_size, 
                size=batch_size)
            batch_x_train = self.x_train[random_batch_ind, ...]
            batch_y_train = self.y_train[random_batch_ind]
            _elbo = -self.sess.run(
                self.nelbo,
                feed_dict={
                    self.data: batch_x_train, 
                    self.labels: batch_y_train,
                    self.training: True
                })
            elbo += _elbo
        elbo /= mc_num
        return elbo

    # Train model
    def train(self, batch_size=200, no_epochs=200, display_epoch=2, 
            x_val=None, y_val=None, verbose=False, test_mc_num=200):
        num_batch = int(np.ceil(1.0 * self.training_size / batch_size))
        sess = self.sess
        training_errs = []
        val_testabserr = []

        # Training cycle
        for epoch in range(no_epochs):
            random_ind = list(range(self.training_size)) 
            self.rng.shuffle(random_ind)
            for batch_id in range(num_batch):
                batch_start = batch_id*batch_size
                batch_end = (batch_id+1)*batch_size
                random_batch_ind = random_ind[batch_start:batch_end]
                batch_x_train = self.x_train[random_batch_ind, ...]
                batch_y_train = self.y_train[random_batch_ind]
                # print_log(batch_x_train.shape, batch_y_train.shape)

                test_abserr = sess.run(
                    self.train_step,
                    feed_dict={
                        self.data: batch_x_train, 
                        self.labels: batch_y_train,
                        self.training: True
                    })

            # Display logs every display_epoch
            if epoch == 0 \
                    or (epoch+1) % display_epoch == 0 \
                    or epoch == no_epochs-1:
                if verbose:
                    train_abserr = self.test_accuracy(
                        self.x_train[:10000], self.y_train[:10000], test_mc_num)
                    print_log(
                        "\ttraining error=", "{:.5f}".format(test_abserr),
                        "training crosss entropy=")
                    if x_val is not None and y_val is not None:
                        val_abserr = self.test_accuracy(
                            x_val, y_val, test_mc_num)
                        print_log(
                            "\tvalidation error=", "{:.5f}".format(val_abserr))
                        val_testabserr.append(val_abserr)
                  
                    training_errs.append(train_abserr)

        print_log("Optimization Finished!")
        return training_errs, val_testabserr

def get_bayesian_neural_net_with_prior(
        prior_m=[None, None, None, None, None, None, None], 
        prior_s=[None, None, None, None, None, None, None],
        initial_prior_var=1.,):
    """
    Args:
    prior_m -- A list of prior mean
    prior_s -- A list of prior standard deviation
    initial_prior_var -- Initial prior variance is set if either prior_m or 
        prior_s is None.

    Returns:
    A tensorflow keras Sequential object as a model architecture.
    """
    kwargs = {
        'kernel_posterior_tensor_fn': (
            lambda d: d.sample()),
    }
    neural_net = tf.keras.Sequential([
        tfp.layers.DenseFlipout(100,
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
            loc_initializer=(lambda *args, **kwargs: prior_m[0]) if prior_m[0] is not None
                            else tf.initializers.random_normal(stddev=0.1),
            untransformed_scale_initializer=(lambda *args, **kwargs: tf.log(tf.exp(prior_s[0])-1)) if prior_s[0] is not None
                                                else tf.initializers.random_normal(mean=-3., stddev=0.1)),
            kernel_prior_fn=ind_multivariate_normal_fn(
            prior_var=initial_prior_var, mu=prior_m[0], sigma=prior_s[0]),
            **kwargs),
        tf.keras.layers.Activation(tf.nn.relu),
        tfp.layers.DenseFlipout(1,
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
            loc_initializer=(lambda *args, **kwargs: prior_m[1]) if prior_m[1] is not None
                            else tf.initializers.random_normal(stddev=0.1),
            untransformed_scale_initializer=(lambda *args, **kwargs: tf.log(tf.exp(prior_s[1])-1)) if prior_s[1] is not None
                                                else tf.initializers.random_normal(mean=-3., stddev=0.1)),
            kernel_prior_fn=ind_multivariate_normal_fn(
            prior_var=initial_prior_var, mu=prior_m[1], sigma=prior_s[1]),
            **kwargs)
        ])
    
    
            

    return neural_net


class BayesianNN(NoLongerAnImageClassificationModel):
    '''
    '''
    def __init__(self, x_train, y_train, neural_net,
            mc_sample=None, learning_rate=0.00001, beta = 1., gamma = 1., rng=None):
        '''Note that `mc_sample` does not work. 
        '''
        del mc_sample # does not work
        super().__init__(x_train, y_train, neural_net, learning_rate, beta, gamma, rng)

    def get_weights(self):
        names = [layer.name for layer in self.neural_net.layers
            if 'flipout' in layer.name or 'dropout' in layer.name]
        # print("***")
        # print(f'qdr: {tf.keras.backend.get_value(qdr[0])}')
        # print("***")
        qm = [layer.kernel_posterior.mean()
            for layer in self.neural_net.layers
            if 'flipout' in layer.name]
        qs = [layer.kernel_posterior.stddev()
            for layer in self.neural_net.layers
            if 'flipout' in layer.name]
        qm_vals, qs_vals = self.sess.run([qm, qs])
        return qm_vals, qs_vals, names



if __name__ == '__main__':

    tf.reset_default_graph()
    random_seed = 1
    rng = np.random.RandomState(seed=random_seed)
    tf.compat.v1.set_random_seed(rng.randint(2**31))
    rng_for_model = np.random.RandomState(seed=random_seed)

    max_iter = 100
    changerate = 3
    task_size = 20000
    independent = False

    initial_prior_var = 1.0 
    lr = 0.0001

    folder_name = './test_case_vcl/'
    sys.stdout = open(folder_name + 'log.txt', 'w')

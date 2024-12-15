#from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers
from tensorflow.keras.layers import Layer

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class DenseVariational(Layer):
    def __init__(self,
                 units,
                 var_weight,
                 prior_params,
                 activation=None,
                 **kwargs):


        self.units = units
        self.var_weight = var_weight
        self.activation = activations.get(activation)
        self.prior_sigma_1 = prior_params['prior_sigma_1']
        self.prior_sigma_2 = prior_params['prior_sigma_2']
        self.prior_pi_1 = prior_params['prior_pi']
        self.prior_pi_2 = 1.0 - self.prior_pi_1
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2)

        super().__init__(**kwargs)



    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units



def f(x, sigma):
    epsilon = np.random.randn(*x.shape) * sigma
    return 10 * np.sin(2 * np.pi * (x)) + epsilon


def gen_data(train_size=32, noise=1.0, show=False):
    X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1)
    y = f(X, sigma=noise)
    y_true = f(X, sigma=0.0)

    if show:
        plt.scatter(X, y, marker='+', label='Training data')
        plt.plot(X, y_true, label='Truth')
        plt.title('Noisy training data and ground truth')
        plt.legend();
        plt.show()
    return X, y, y_true


train_size = 32
noise = 1.0

def neg_log_likelihood(y_obs, y_pred, sigma=noise):
    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    return tf.math.reduce_sum(-dist.log_prob(y_obs))


if __name__ == "__main__":
    gen_data(32, 1.0, True)



batch_size = train_size
num_batches = train_size / batch_size

var_weight = 1.0 / num_batches
prior_params = {
    'prior_sigma_1': 1.5,
    'prior_sigma_2': 0.1,
    'prior_pi': 0.5
}

x_in = Input(shape=(1,))
x = DenseVariational(20, var_weight, prior_params, activation='relu')(x_in)
x = DenseVariational(20, var_weight, prior_params, activation='relu')(x)
x = DenseVariational(1, var_weight, prior_params)(x)

model = Model(x_in, x)

if int(tfv[1]) < 11:
    model.compile(loss=neg_log_likelihood, optimizer=optimizers.Adam(lr=0.08), metrics=['mse'])
else:
    model.compile(loss=neg_log_likelihood, optimizer=optimizers.Adam(learning_rate=0.08), metrics=['mse'])

X, y, _ = gen_data(train_size, noise, False)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

print('start fitting the model....')
model.fit(X, y, batch_size=batch_size, epochs=1500, verbose=1, callbacks=[tensorboard_callback])

############################################################
# Testing
############################################################

X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
y_pred_list = []

for i in tqdm.tqdm(range(500)):
    y_pred = model(X_test, training=False)  # model.predict(X_test)
    y_pred_list.append(y_pred.numpy())

y_preds = np.concatenate(y_pred_list, axis=1)

y_mean = np.mean(y_preds, axis=1)
y_sigma = np.std(y_preds, axis=1)

plt.plot(X_test, y_mean, 'r-', label='Predictive mean')
plt.scatter(X, y, marker='+', label='Training data')
plt.fill_between(X_test.ravel(),
                 y_mean + 2 * y_sigma,
                 y_mean - 2 * y_sigma,
                 alpha=0.3, label='Epistemic uncertainty',
                 color='yellow')
plt.title('Prediction')
plt.legend()
plt.savefig("vi_train_test.pdf")
plt.show()


    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        self.add_loss(self.var_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.var_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(tf.linalg.matmul(inputs, kernel) + bias)

    def var_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.var_weight * tf.math.reduce_sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

    def log_prior_prob(self, w):tf_ver = tf.__version__
tfv = tf_ver.split('.')
print(int(tfv[1]))
if int(tfv[0]) < 2:
    print('Need tensorflow version 2')
    exit(0)
        comp_1_dist = tfp.distributions.Normal(0.0, self.prior_sigma_1)
        comp_2_dist = tfp.distributions.Normal(0.0, self.prior_sigma_2)
        return tf.math.log(self.prior_pi_1 * comp_1_dist.prob(w) +
                     self.prior_pi_2 * comp_2_dist.prob(w))

batch_size = train_size
num_batches = train_size / batch_size

var_weight = 1.0 / num_batches
prior_params = {
    'prior_sigma_1': 1.5,
    'prior_sigma_2': 0.1,
    'prior_pi': 0.5
}

x_in = Input(shape=(1,))
x = DenseVariational(20, var_weight, prior_params, activation='relu')(x_in)
x = DenseVariational(20, var_weight, prior_params, activation='relu')(x)
x = DenseVariational(1, var_weight, prior_params)(x)

model = Model(x_in, x)

if (int(tfv[1]) <11):
    model.compile(loss=neg_log_likelihood, optimizer=optimizers.Adam(lr=0.08), metrics=['mse'])
else:
    model.compile(loss=neg_log_likelihood, optimizer=optimizers.Adam(learning_rate=0.08), metrics=['mse'])

X,y,_= gen_data(train_size,noise, False)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")


print('start fitting the model....')
model.fit(X, y, batch_size=batch_size, epochs=1500, verbose=1, callbacks=[tensorboard_callback]);



############################################################
# Testing
############################################################

X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
y_pred_list = []


for i in tqdm.tqdm(range(500)):
    y_pred = model(X_test, training=False) #model.predict(X_test)
    y_pred_list.append(y_pred.numpy())

y_preds = np.concatenate(y_pred_list, axis=1)

y_mean = np.mean(y_preds, axis=1)
y_sigma = np.std(y_preds, axis=1)


plt.plot(X_test, y_mean, 'r-', label='Predictive mean');
plt.scatter(X, y, marker='+', label='Training data')
plt.fill_between(X_test.ravel(),
                 y_mean + 2 * y_sigma,
                 y_mean - 2 * y_sigma,
                 alpha=0.3, label='Epistemic uncertainty',
                 color='yellow')
plt.title('Prediction')
plt.legend();
plt.savefig("vi_train_test.pdf")
plt.show()
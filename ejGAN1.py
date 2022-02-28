"""
Author: Jamal Toutouh (www.jamal.es)

It contains the code to show an example about using Generative Adversarial Networks.
The GAN is used to generate vectors of a given size that contains float numbers that follow a given Normal distribution
defined my the mean and standard deviation.

Spanish: Ejemplo en el que se entrena una GAN para que cree vectores con valores aleatorios que cumplan con una
distribución normal dada la media y la desviación estandar


Original file is located at
    https://colab.research.google.com/drive/1gbTlefMoY6eQDlZXpCU9PVINo55Yb3ly
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.autograd import Variable, autograd

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import IsotonicRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

from pandas import read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler
import random

# """Definimos los datos reales: distribución normal de media *real_data_mean* y desviación estándar *real_data_stddev*"""

# # Real data set: Samples from a normal distribution given real_data_mean and real_data_sttdev
# real_data_mean = 4.0
# real_data_stddev = 0.2


# def plot_data(real_mean, real_sigma, fake_data=None, epoch=None):
#     x = np.linspace(1, 7, 100)
#     plt.figure()

#     real_data_distribution = norm.pdf(x, real_mean, real_sigma)
#     plt.plot(x, real_data_distribution, 'b', linewidth=2, label='Real data')
    
#     if not (fake_data is None):
#       mu, std = norm.fit(fake_data.tolist())
#       fake_data_distribution = norm.pdf(x, mu, std)
#       plt.plot(x, fake_data_distribution, 'k', linewidth=2, label='Fake data')
#       title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
#     else:
#       title = "Real data: mu = %.2f,  std = %.2f" % (real_mean, real_sigma)
#       epoch = "real"
#     plt.xlim(2,6)
#     plt.ylim(0, 2.5)
#     plt.title(title)
#     plt.legend()
#     plt.savefig('01-1d-normal-gan-data-{}.png'.format(epoch))
#     plt.show()

# def get_real_sampler(mu, sigma):
#     """
#     Creates a lambda function to create samples of the real data (Normal) distribution
#     :param mu: Mean of the real data distribution
#     :param sigma: Standard deviation of the real data distribution
#     :return: Lambda function sampler
#     """
#     dist = Normal(mu,sigma )
#     return lambda m, n: dist.sample((m, n)).requires_grad_()


# # Load samples from real data
# get_real_data = get_real_sampler(real_data_mean, real_data_stddev)
# plot_data(real_data_mean, real_data_stddev)


"""Función que crea los vectores del espacio latente que leerá el generador para crear los datos"""


# def read_latent_space(batch_size, latent_vector_size):
    """
    Creates a tensor with random values fro latent space  with shape = size
    :param size: Size of the tensor (batch size).
    :return: Tensor with random values (z) with shape = size
    """
    z = torch.rand(batch_size, latent_vector_size)
    if torch.cuda.is_available():
        return z.cuda()
    return z


class Generator(nn.Module):
    """
    Class that defines the the Generator Neural Network
    """

    def __init__(self, window_size, hidden_size, feature_size):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.LSTM(input_size=(window_size, 1), hidden_size=hidden_size, num_layers=2),
            nn.GELU(),
            # nn.Linear(in_features=K * c, out_features=K * F),
            nn.Linear(in_features=(window_size, hidden_size), out_features=(window_size, feature_size)),
            nn.GELU(),
        )

        # self.input_lstm_layer = nn.LSTM(input_size=K, hidden_size=c, num_layers=2)
        # # self.lstm_activation = nn.GELU()
        # self.dense_layer = nn.Linear(in_features=K * c, out_features=K * F)
        # # self.dense_activation = nn.GELU()

    def forward(self, x):
        x = self.net(x)
        # x, _, _ = self.input_lstm_layer(x)
        # x = nn.GELU(x)
        # # x = self.lstm_activation(x)
        # x = self.dense_layer(x)
        # # x = self.dense_activation(x)
        # x = nn.GELU(x)
        return x


class Discriminator(nn.Module):
    """
    Class that defines the the Discriminator Neural Network
    """

    def __init__(self, window_size, hidden_size, feature_size):
        super(Discriminator, self).__init__()
        self.window_size = window_size

        self.net = nn.Sequential(
            nn.LSTM(input_size=(window_size, feature_size), hidden_size=hidden_size, num_layers=2),
            nn.GELU(),
            nn.Linear(in_features=(window_size, hidden_size), out_features=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.net(x)
        return x

    # MH-GAN
    def calibrate_discriminator(self, X, X_fake):
        # First generating an equal number of samples
        self.calibrator = IsotonicRegression(0, 1)
        # g_samples = self.generate_samples(X.shape[0])
        # g_samples = self.generate_samples(X.shape[0])
        inputs = torch.cat((X, X_fake), 0)
        # preds = self.score_samples(inputs).numpy().flatten()
        preds = self.forward(inputs)  # ???
        # y = np.array([1]*X.shape[0] + [0]*X.shape[0])
        y = torch.cat((torch.ones(X.shape[0]), torch.zeros(X_fake.shape[0])), 0)
        self.calibrator.fit(preds, y)

    def calibrated_score_samples(self, X):
        # s = self.score_samples(X).numpy()[0]
        s = self.forward(X)  # ???
        return self.calibrator.transform(s)

    # def generate_mh_samples(self, num_samples, init_sample):
    #     # Creating our initial real sample
    #     y = tf.expand_dims(init_sample, axis=0)
    #     y_score = self.calibrated_score_samples(y)[0]

    #     # Creating lists to track samples
    #     samples = []
    #     while len(samples) < num_samples:

    #         # Sampling a random vector and getting G
    #         x = self.generate_samples(1).numpy()

    #         # Calculating MC prediction
    #         x_score = self.calibrated_score_samples(x)[0]
    #         x = x[0]

    #         # Now testing for acceptance
    #         u = np.random.uniform(0, 1, (1,))[0]
    #         if u <= np.fmin(1., (1./y_score - 1.)/(1./x_score - 1.)):
    #             y = x
    #             y_score = x_score
    #             samples.append(x)

    #     return np.stack(samples)


def generate_mh_samples(D, G, sample_prefixes, init_sample):
    # Creating our initial real sample
    y_score = D.calibrated_score_samples(init_sample)

    res = torch.zeros(sample_prefixes.shape[0], sample_prefixes.shape[1], sample_prefixes.shape[2] + 1)

    for idx, sample_prefix in enumerate(sample_prefixes):

        # sample latent space and generate fake
        z = torch.randn(sample_prefix.shape[0], 1)
        z = torch.cat((sample_prefix, z), 0)
        x_fake = G(z)

        # Calculating MC prediction
        x_score = D.calibrated_score_samples(x_fake)

        for i in range(500):
            # Now testing for acceptance
            u = np.random.uniform(0, 1)
            if u <= min(1, (1 / y_score - 1) / (1 / x_score - 1)):
                y_score = x_score
                res[idx, :, :] = x_fake
                break
        else:
            print('ERROR')

    return res



# """Función auxiliar para mostrar la evolución de la función de pérdida del generador y el discriminador"""

# def plot_loss_evolution(discriminator_loss, generator_loss):
#     x = range(len(discriminator_loss)) if len(discriminator_loss) > 0 else range(len(generator_loss))
#     if len(discriminator_loss) > 0: plt.plot(x, discriminator_loss, '-b', label='Discriminator loss')
#     if len(generator_loss) > 0: plt.plot(x, generator_loss, ':r', label='Generator loss')
#     plt.legend()
#     plt.savefig('01-1d-normal-gan-loss.png')
#     plt.show()


# """Funciones para crear las etiquetas de dato real *real_data_target* y dato falso *fake_data_target* que emplea el discriminador para calcular la función de pérdida"""

# def real_data_target(size):
#     """
#     Creates a tensor with the target for real data with shape = size
#     :param size: Size of the tensor (batch size).
#     :return: Tensor with real label value (ones) with shape = size
#     """
#     data = Variable(torch.ones(size, 1))
#     if torch.cuda.is_available(): return data.cuda()
#     return data

# def fake_data_target(size):
#     """
#     Creates a tensor with the target for fake data with shape = size
#     :param size: Size of the tensor (batch size).
#     :return: Tensor with fake label value (zeros) with shape = size
#     """
#     data = Variable(torch.zeros(size, 1))
#     if torch.cuda.is_available(): return data.cuda()
#     return data


def main():

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get DATA
    data = read_csv('C:/Users/D255728/Documents/ProyectoGAN/datos/datos_diarios_GAN.csv', header=0, index_col=0, encoding='latin-1')

    # categorical to numeric
    data['E4'] = data['E4'].astype('category')
    data['E4'] = data['E4'].cat.codes
    data['tipo_dia'] = data['tipo_dia'].astype('category')
    data['tipo_dia'] = data['tipo_dia'].cat.codes

    # normalize data
    data[data.columns] = MinMaxScaler().fit_transform(data[data.columns])
    data = torch.tensor(data.values).to(DEVICE)

    # split train/test
    train_data = data[0:3287]
    test_data = data[3287:]

    K = 20  # window size
    S = 1  # step size
    F = train_data.shape[-1]  # features
    c = 128  # cell state size

    p_coeff = 10  # lambda in GP
    step = 0
    n_critic = 5

    batch_size = 99

    # Creating the GAN generator
    G = Generator(window_size=K, hidden_size=c, feature_size=F)
    generator_learning_rate = 2e-6
    # generator_loss = nn.BCELoss()
    G_opt = optim.Adam(G.parameters(), lr=generator_learning_rate)
    # G_opt = optim.Adam(G.parameters(), lr=generator_learning_rate, betas=(0., 0.9))

    # Creating the GAN discriminator
    D = Discriminator(window_size=K, hidden_size=c, feature_size=F)

    discriminator_learning_rate = 2e-6
    # discriminator_loss = nn.BCELoss()
    D_opt = optim.Adam(D.parameters(), lr=discriminator_learning_rate)
    # D_opt = optim.Adam(D.parameters(), lr=discriminator_learning_rate, betas=(0., 0.9))

    epochs = 1500  # Training epochs

    print(f'Starting adversarial GAN training for {epochs} epochs.')

    for epoch in range(epochs):

        # batch shape: (batch_size, K, F)

        idxs = list(range(1, 3288))  # solo para S == 1
        random.shuffle(idxs)
        # ===================================================================================

        while len(idxs) >= batch_size:

            D.zero_grad()
            # Entrenar el crítico

            # Datos reales
            x = torch.zeros(batch_size, K, F).to(DEVICE)
            for i in range(batch_size):
                idx = idxs.pop()
                x[i, :, :] = train_data.loc[idx:idx + K - 1, :]

            # Muestreo y generación
            z = torch.randn(batch_size, K, 1).to(DEVICE)
            x_fake = G(z)

            # Penalización de gradientes (e.g. gradientes respecto a x_penalty)
            eps = torch.rand(batch_size, 1, 1).to(DEVICE)  # x shape: (batch_size, K, F)
            x_penalty = eps * x + (1 - eps) * x_fake
            x_penalty = x_penalty.view(x_penalty.size(0), -1)
            p_outputs = D(x_penalty)
            # Calcular la suma de gradientes de salidas (outputs) respecto a las entradas (inputs)
            # xp_grad = autograd.grad(outputs=p_outputs, inputs=x_penalty, grad_outputs=C_labels,
            #                         create_graph=True, retain_graph=True, only_inputs=True)
            xp_grad = autograd.grad(outputs=p_outputs, inputs=x_penalty, create_graph=True)
            grad_penalty = p_coeff * torch.mean(torch.pow(torch.norm(xp_grad[0], 2, 1) - 1, 2))

            # Wasserstein loss
            x_outputs = D(x)
            z_outputs = D(x_fake)
            D_x_loss = torch.mean(x_outputs)
            D_z_loss = torch.mean(z_outputs)
            D_loss = D_z_loss - D_x_loss + grad_penalty

            D_loss.backward()
            D_opt.step()

            if step % n_critic == 0:
                D.zero_grad()
                G.zero_grad()
                # Entrenar el generador
                z = torch.randn(batch_size, K, 1).to(DEVICE)
                z_outputs = D(G(z))
                G_loss = -torch.mean(z_outputs)

                G_loss.backward()
                G_opt.step()

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}/{epochs}, Step: {step}, D Loss: {D_loss.item()}, G Loss: {G_loss.item()}')
                # print('Epoch: {}/{}, Step: {}, C Loss: {}, G Loss: {}'.format(epoch, epochs, step, D_loss.item(), G_loss.item()))

            # if step % 1000 == 0:
            #     G.eval()
            #     img = get_sample_image(G, n_noise)
            #     imsave('samples/{}_step{:05d}.jpg'.format(MODEL_NAME, step), img, cmap='gray')
            #     G.train()
            step += 1
        # ===================================================================================

    torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/generator.pth')
    torch.save(D.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/discriminator.pth')

    # generate 2019 data
    real_2019 = test_data['Demanda']
    predicted_2019 = generate_mh_samples(D, G, test_data[:, :-1], train_data.sample)[:, -1]

    DataFrame(predicted_2019.numpy()).to_csv('C:/Users/D255728/Documents/ProyectoGAN/datos/predicted_2019.csv')
    rmse = sqrt(mean_squared_error(real_2019, predicted_2019))
    print(f'{rmse=}')


# """Función principal que define el entrenamiento de la GAN"""

# def main():
#     vector_length = 10  # Defines the length of the vector that defines a data sample

#     generator_input_size = 50  # Input size of the generator (latent space)
#     generator_hidden_size = 150
#     generator_output_size = vector_length

#     discriminator_input_size = vector_length
#     discriminator_hidden_size = 75
#     discriminator_output_size = 1

#     batch_size = 30
#     number_of_batches = 600

#     # Creating the GAN generator
#     generator = Generator(input_size=generator_input_size, hidden_size=generator_hidden_size,
#                           output_size=generator_output_size)
#     generator_learning_rate = 0.0008
#     generator_loss = nn.BCELoss()
#     generator_optimizer = optim.SGD(generator.parameters(), lr=generator_learning_rate, momentum=0.9)

#     # Creating the GAN discriminator
#     discriminator = Discriminator(input_size=discriminator_input_size, hidden_size=discriminator_hidden_size,
#                                   output_size=discriminator_output_size)

#     discriminator_learning_rate = 0.001
#     discriminator_loss = nn.BCELoss()
#     discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=discriminator_learning_rate, momentum=0.9)

#     epochs = 20 # Training epochs
#     freeze_generator_steps = 1

#     noise_for_plot = read_latent_space(batch_size, generator_input_size)
#     discriminator_loss_storage, generator_loss_storage = [], []

#     print('Real dataset loaded...')
#     print('Starting adversarial GAN training for {} epochs.'.format(epochs))

#     # Plot a little bit of trash
#     generator_output = generator(noise_for_plot)
#     plot_data(real_data_mean, real_data_stddev, generator_output, -1)

#     for epoch in range(epochs):

#         batch_number = 0

#         # training discriminator
#         while batch_number < number_of_batches:  

#             # 1. Train the discriminator
#             discriminator.zero_grad()
#             # 1.1 Train discriminator on real data
#             input_real = get_real_data(batch_size, discriminator_input_size)
#             discriminator_real_out = discriminator(input_real)
#             discriminator_real_loss = discriminator_loss(discriminator_real_out, real_data_target(batch_size))
#             discriminator_real_loss.backward()
#             # 1.2 Train the discriminator on data produced by the generator
#             input_fake = read_latent_space(batch_size, generator_input_size)
#             generator_fake_out = generator(input_fake).detach() # Synthetic data
#             discriminator_fake_out = discriminator(generator_fake_out)
#             discriminator_fake_loss = discriminator_loss(discriminator_fake_out, fake_data_target(batch_size))
#             discriminator_fake_loss.backward()
#             # 1.3 Optimizing the discriminator weights
#             discriminator_optimizer.step()

#             # discriminator_total_loss = discriminator_real_loss + discriminator_fake_loss
#             # discriminator_total_loss.backward()
#             # discriminator_optimizer.step()


#             # 2. Train the generator
#             if batch_number % freeze_generator_steps == 0:
#               generator.zero_grad()
#               # 2.1 Create fake data
#               input_fake = read_latent_space(batch_size, generator_input_size)
#               generator_fake_out = generator(input_fake)
#               # 2.2 Try to fool the discriminator with fake data
#               discriminator_out_to_train_generator = discriminator(generator_fake_out)
#               discriminator_loss_to_train_generator = generator_loss(discriminator_out_to_train_generator,
#                                                                     real_data_target(batch_size))
#               discriminator_loss_to_train_generator.backward()
#               # 2.3 Optimizing the generator weights
#               generator_optimizer.step()

#             batch_number += 1

#         discriminator_loss_storage.append(discriminator_fake_loss + discriminator_real_loss)
#         generator_loss_storage.append(discriminator_loss_to_train_generator)
#         print('Epoch={}, Discriminator loss={}, Generator loss={}'.format(epoch, discriminator_loss_storage[-1],
#                                                                           generator_loss_storage[-1]))

#         if epoch % 1 == 0:
#             generator_output = generator(noise_for_plot)
#             plot_data(real_data_mean, real_data_stddev, generator_output, epoch)


#     plot_loss_evolution(discriminator_loss_storage, generator_loss_storage)

main()

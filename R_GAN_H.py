import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch import autograd

import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

from pandas import read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler
import random
import time
import sys


class Generator(nn.Module):
    """
    Class that defines the the Generator Neural Network
    """

    def __init__(self, window_size, hidden_size, feature_size):
        super(Generator, self).__init__()

        # self.input_lstm_layer = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)  # Z_1
        self.input_lstm_layer = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=2, batch_first=True)  # Z_1
        # self.input_lstm_layer = nn.LSTM(input_size=window_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.lstm_activation = nn.GELU()
        self.dense_layer = nn.Linear(in_features=window_size * hidden_size, out_features=window_size * feature_size)
        self.dense_activation = nn.GELU()

    def forward(self, x):
        x, (_, _) = self.input_lstm_layer(x)
        x = self.lstm_activation(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense_layer(x)
        x = self.dense_activation(x)
        return x


class Discriminator(nn.Module):
    """
    Class that defines the the Discriminator Neural Network
    """

    def __init__(self, window_size, hidden_size, feature_size):
        super(Discriminator, self).__init__()
        self.window_size = window_size

        # self.input_lstm_layer = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.input_lstm_layer = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # self.input_lstm_layer = nn.LSTM(input_size=window_size*feature_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.lstm_activation = nn.GELU()
        self.dense_layer = nn.Linear(in_features=window_size * hidden_size, out_features=1)
        self.dense_activation = nn.Tanh()

    def forward(self, x):
        x, (_, _) = self.input_lstm_layer(x)
        x = self.lstm_activation(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense_layer(x)
        x = self.dense_activation(x)
        return x

    # MH-GAN
    def calibrate_discriminator(self, X, X_fake):
        self.calibrator = IsotonicRegression(y_min=0, y_max=1)
        inputs = torch.cat((X, X_fake), 0)
        preds = self.forward(inputs)
        y = torch.cat((torch.ones(X.shape[0]), torch.zeros(X_fake.shape[0])), 0)
        self.calibrator.fit(preds.detach().numpy(), y)

    def calibrated_score_samples(self, X):
        s = self.forward(X.double().reshape(1, X.shape[0], X.shape[1]))
        return self.calibrator.transform(s.detach().numpy())


def generate_mh_samples(G, D, num_samples, init_sample):
    # Creating our initial real sample
    y = init_sample
    y_score = D.calibrated_score_samples(init_sample)

    # Creating lists to track samples
    samples = []
    i = 0
    # print(f'{init_sample.shape=}')
    while len(samples) < num_samples:
        if i % 250 == 0:
            print(f'{i=} {len(samples)=}')
        i += 1

        # Sampling a random vector and getting G
        z = torch.randn(1, init_sample.shape[0], 1)
        # x = G(z)
        x = G(z).reshape(init_sample.shape[0], init_sample.shape[1])
        # print(f'{x.shape=}')

        # Calculating MC prediction
        x_score = D.calibrated_score_samples(x)[0]
        # x = x[0]

        if torch.sum(x[:, -1] < 0) + torch.sum(x[:, -1] > 1) > 0:
            continue

        # Now testing for acceptance
        u = np.random.uniform(0, 1, (1,))[0]
        if u <= np.fmin(1., (1. / y_score - 1.) / (1. / x_score - 1.)):
            y = x
            y_score = x_score
            samples.append(x.detach().numpy())

    return np.stack(samples)


def main(win_size=24, step_size=12, eps=3000, n_crit=5):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get DATA
    data = read_csv('C:/Users/D255728/Documents/ProyectoGAN/datos/datos_horarios_GAN.csv', header=0, index_col=0, sep=';', decimal=',', encoding='latin-1')
    # data = read_csv('C:/Users/D255728/Documents/ProyectoGAN/datos/test.csv', index_col=0)

    # # categorical to numeric
    # data['tipo_dia'] = data['tipo_dia'].astype('category')
    # data['tipo_dia'] = data['tipo_dia'].cat.codes

    data = data[['Mes', 'Dia', 'Hora', 'semana', 'Temperatura', 'arima', 'Demanda']]

    # normalize data
    data[data.columns] = MinMaxScaler().fit_transform(data[data.columns])
    original_data = data.copy()
    data = torch.tensor(data.values).to(DEVICE)

    # split train/test
    split_idx = 78888
    train_data = data[0:split_idx]
    test_data = data[split_idx:]

    K = win_size  # window size
    S = step_size  # step size
    F = train_data.shape[-1]  # features
    c = 128  # cell state size

    # p_coeff = 10  # lambda en GP
    step = 0
    n_critic = n_crit

    batch_size = 99

    epochs = eps

    dynamic_LR = False

    # Creating the GAN generator
    G = Generator(window_size=K, hidden_size=c, feature_size=F).to(DEVICE)
    generator_learning_rate = 2e-6
    # G_opt = optim.Adam(G.parameters(), lr=generator_learning_rate)
    G_opt = optim.Adam(G.parameters(), lr=generator_learning_rate, betas=(0., 0.9))
    if dynamic_LR:
        G_scheduler = optim.lr_scheduler.LambdaLR(G_opt, lr_lambda=(lambda epoch: 1 / (1 + round(epoch / epochs))))

    # Creating the GAN discriminator
    D = Discriminator(window_size=K, hidden_size=c, feature_size=F).to(DEVICE)

    discriminator_learning_rate = 2e-6
    # D_opt = optim.Adam(D.parameters(), lr=discriminator_learning_rate)
    D_opt = optim.Adam(D.parameters(), lr=discriminator_learning_rate, betas=(0., 0.9))
    if dynamic_LR:
        D_scheduler = optim.lr_scheduler.LambdaLR(D_opt, lr_lambda=(lambda epoch: 1 / (1 + round(epoch / epochs))))

    print(f'Starting adversarial GAN training for {epochs} epochs.')

    start_time = time.time()

    best_D_values = 100

    for epoch in range(1, epochs + 1):

        D_x_value = []
        D_z_value = []

        idxs = list(range(0, split_idx - K, S))
        random.shuffle(idxs)

        while len(idxs) >= batch_size:

            D.zero_grad()
            # Entrenar el crítico

            # Datos reales
            x = torch.zeros(batch_size, K, F).to(DEVICE)
            for i in range(batch_size):
                idx = idxs.pop()
                x[i, :, :] = train_data[idx:idx + K, :]

            # Muestreo y generación
            z = torch.randn(batch_size, K, 1).to(DEVICE)  # Z_1
            x_fake = G(z)
            x_fake = x_fake.reshape(x.shape)

            # Gradient Penalty - (GP)
            # eps = torch.rand(batch_size, 1, 1).to(DEVICE)  # x shape: (batch_size, K, F)
            # x_penalty = eps * x + (1 - eps) * x_fake
            # # x_penalty = x_penalty.view(x_penalty.size(0), -1)
            # with torch.backends.cudnn.flags(enabled=False):
            #     p_outputs = D(x_penalty)
            # grad_outputs = torch.ones_like(p_outputs)
            # xp_grad = autograd.grad(outputs=p_outputs, inputs=x_penalty, grad_outputs=grad_outputs, create_graph=True)
            # grad_penalty = p_coeff * torch.mean(torch.pow(torch.norm(xp_grad[0].reshape(batch_size, -1), 2, 1) - 1, 2))  # !!! norma en dim 1 luego del resize

            # Wasserstein loss
            x_outputs = D(x)
            z_outputs = D(x_fake)
            D_x_loss = torch.mean(x_outputs)
            D_z_loss = torch.mean(z_outputs)
            # D_loss = D_z_loss - D_x_loss + grad_penalty
            D_loss = D_z_loss - D_x_loss


            ####### DEBUG ########
            D_x_value.append(D_x_loss.item())
            D_z_value.append(D_z_loss.item())
            ####### DEBUG ########

            D.zero_grad()
            D_loss.backward()
            D_opt.step()
            if dynamic_LR:
                D_scheduler.step()

            # Poda de pesos para la restricción K-Lipshitziana
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            if step % n_critic == 0:
                D.zero_grad()
                G.zero_grad()
                # Entrenar el generador
                z = torch.randn(batch_size, K, 1).to(DEVICE)  # Z_1
                z_outputs = D(G(z).reshape(x.shape))
                G_loss = -torch.mean(z_outputs)

                G_loss.backward()
                G_opt.step()
                if dynamic_LR:
                    G_scheduler.step()

            step += 1

        if epoch == 500:
            torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_500' + '.pth')

        if epoch == 1000:
            torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_1000' + '.pth')

        if epoch == 1500:
            torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_1500' + '.pth')

        if epoch == 2000:
            torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_2000' + '.pth')

        if epoch == 2500:
            torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_2500' + '.pth')

        if epoch == 3000:
            torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_3000' + '.pth')

        if epoch == 3500:
            torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_3500' + '.pth')

        if epoch == 4000:
            torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_4000' + '.pth')

        if epoch == 4500:
            torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_4500' + '.pth')

        if epoch == 5000:
            torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_5000' + '.pth')

        if epoch == 5500:
            torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_5500' + '.pth')

        # if 250 <= epoch < 1000:
        #     if best_D_values > np.abs(np.mean(D_z_value) - np.mean(D_x_value)):
        #         best_D_values = np.abs(np.mean(D_z_value) - np.mean(D_x_value))
        #         torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_best_1000' + '.pth')
        #         # torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/1_layer_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_best_1000' + '.pth')
        # if epoch == 1000:
        #     print(f'{best_D_values=}')
        #     best_D_values = 100
        # if 1000 <= epoch < 2000:
        #     if best_D_values > np.abs(np.mean(D_z_value) - np.mean(D_x_value)):
        #         best_D_values = np.abs(np.mean(D_z_value) - np.mean(D_x_value))
        #         torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_best_2000' + '.pth')
        #         # torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/1_layer_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_best_2000' + '.pth')
        # if epoch == 2000:
        #     print(f'{best_D_values=}')
        #     best_D_values = 100
        # if 2000 <= epoch:
        #     if best_D_values > np.abs(np.mean(D_z_value) - np.mean(D_x_value)):
        #         best_D_values = np.abs(np.mean(D_z_value) - np.mean(D_x_value))
        #         torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_best_3000' + '.pth')
        #         # torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/1_layer_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '_best_3000' + '.pth')

        if epoch % 1 == 0:
            # print(f'Epoch: {epoch}/{epochs} D_Loss: {D_loss.item():.4f} D_z: {np.mean(D_z_value):.4f}±{np.var(D_z_value):.4f} D_x: {np.mean(D_x_value):.4f}±{np.var(D_x_value):.4f} GP: {grad_penalty:.4f} G_Loss: {G_loss.item():.4f} Time: {(time.time() - start_time)/60:.1f}m')
            print(f'Epoch: {epoch}/{epochs} D_Loss: {D_loss.item():.4f} D_z: {np.mean(D_z_value):.4f}±{np.var(D_z_value):.4f} D_x: {np.mean(D_x_value):.4f}±{np.var(D_x_value):.4f} G_Loss: {G_loss.item():.4f} Time: {(time.time() - start_time)/60:.1f}m')

    print(f'{best_D_values=}')

    torch.save(G.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '.pth')
    torch.save(D.state_dict(), 'C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_discriminator_R_H_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '_' + str(n_critic) + '.pth')


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

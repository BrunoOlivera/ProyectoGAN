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

        self.input_lstm_layer = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=2, batch_first=True)  # Z_n
        # self.input_lstm_layer = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=2, batch_first=True)  # Z_1
        self.lstm_activation = nn.GELU()
        # self.dense_layer = nn.Linear(in_features=window_size * hidden_size, out_features=window_size * feature_size)
        self.dense_layer = nn.Linear(in_features=window_size * hidden_size, out_features=window_size * 1)  # C-GAN
        self.dense_activation = nn.GELU()

    def forward(self, x, y):
        # print(f'{x.shape=}')
        # print(f'{y.shape=}')
        x = torch.cat((y, x), 2)
        # print(f'{x.shape=}')
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

    # def __init__(self, window_size, hidden_size, feature_size):
    def __init__(self, window_size, hidden_size, feature_size):
        super(Discriminator, self).__init__()
        self.window_size = window_size

        self.input_lstm_layer = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.lstm_activation = nn.GELU()
        self.dense_layer = nn.Linear(in_features=window_size * hidden_size, out_features=1)
        self.dense_activation = nn.Tanh()

    def forward(self, x, y):
        # print(f'{x.shape=}')
        # print(f'{y.shape=}')
        x = torch.cat((y, x), 2)
        # print(f'{x.shape=}')
        x, (_, _) = self.input_lstm_layer(x)
        x = self.lstm_activation(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense_layer(x)
        x = self.dense_activation(x)
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

def main(win_size=20, step_size=1, eps=3000):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('DEVICE' + str(DEVICE))

    # Get DATA
    # data = read_csv('C:/Users/D255728/Documents/ProyectoGAN/datos/datos_diarios_GAN.csv', header=0, index_col=0, encoding='latin-1')
    data = read_csv('datos/datos_diarios_GAN.csv', header=0, index_col=0, encoding='latin-1')

    # categorical to numeric
    data['E4'] = data['E4'].astype('category')
    data['E4'] = data['E4'].cat.codes
    data['tipo_dia'] = data['tipo_dia'].astype('category')
    data['tipo_dia'] = data['tipo_dia'].cat.codes

    # normalize data
    data[data.columns] = MinMaxScaler().fit_transform(data[data.columns])
    original_data = data.copy()
    data = torch.tensor(data.values).to(DEVICE)

    # split train/test
    train_data = data[0:3287]
    test_data = data[3287:]

    K = win_size  # window size
    S = step_size  # step size
    F = train_data.shape[-1]  # features
    # F = 1
    c = 128  # cell state size

    p_coeff = 10  # lambda in GP
    step = 0
    n_critic = 5

    batch_size = 99

    epochs = eps
    # epochs = 6000  # Training epochs
    # epochs = 3000  # Training epochs
    # epochs = 1500  # Training epochs
    # epochs = 2  # Training epochs

    dynamic_LR = False

    # Creating the GAN generator
    G = Generator(window_size=K, hidden_size=c, feature_size=F).to(DEVICE)
    # generator_learning_rate = 1e-6
    # generator_learning_rate = 2e-6
    generator_learning_rate = 2e-5
    # G_opt = optim.Adam(G.parameters(), lr=generator_learning_rate)
    G_opt = optim.Adam(G.parameters(), lr=generator_learning_rate, betas=(0., 0.9))
    if dynamic_LR:
        G_scheduler = optim.lr_scheduler.LambdaLR(G_opt, lr_lambda=(lambda epoch: 1 / (1 + round(epoch / epochs))))

    # Creating the GAN discriminator
    D = Discriminator(window_size=K, hidden_size=c, feature_size=F).to(DEVICE)
    # discriminator_learning_rate = 1e-6
    # discriminator_learning_rate = 2e-6
    discriminator_learning_rate = 2e-5
    # D_opt = optim.Adam(D.parameters(), lr=discriminator_learning_rate)
    D_opt = optim.Adam(D.parameters(), lr=discriminator_learning_rate, betas=(0., 0.9))
    if dynamic_LR:
        D_scheduler = optim.lr_scheduler.LambdaLR(D_opt, lr_lambda=(lambda epoch: 1 / (1 + round(epoch / epochs))))

    print('Starting adversarial GAN training for ' + str(epochs) + ' epochs.')

    start_time = time.time()

    for epoch in range(1, epochs + 1):

        # batch shape: (batch_size, K, F)

        # idxs = list(range(1, 3288-K))  # solo para S == 1
        # idxs = list(range(0, 3287 - K))  # solo para S == 1
        idxs = list(range(0, 3287 - K, S))
        random.shuffle(idxs)
        # ===================================================================================

        while len(idxs) >= batch_size:

            # print(f'{len(idxs)=}')

            D.zero_grad()
            # Entrenar el crítico

            # Datos reales
            # x = torch.zeros(batch_size, K*F).to(DEVICE)
            x = torch.zeros(batch_size, K, F).to(DEVICE)
            for i in range(batch_size):
                idx = idxs.pop()
                # x[i, :] = train_data.loc[idx:idx + K - 1, :].reshape(-1,)
                # print(f'{train_data[idx:idx + K, :].shape=}')
                x[i, :, :] = train_data[idx:idx + K, :]
                # x[i, :] = train_data.loc[idx:idx + K - 1, :]
            # print(f'{x.shape=}')

            # Muestreo y generación
            y = x[:, :, :-1].to(DEVICE)
            x = x[:, :, -1]
            x = x.reshape(batch_size, K, 1)
            # z = torch.randn(batch_size, K, K).to(DEVICE)
            z = torch.randn(batch_size, K, 1).to(DEVICE)  # Z_1
            # z = torch.randn(batch_size, K, F).to(DEVICE)  # Z_n
            # print(f'{z.shape=}')
            x_fake = G(z, y)
            # print(f'{x_fake.shape=}')
            x_fake = x_fake.reshape(x.shape)
            # print(f'{x_fake.shape=}')

            # Penalización de gradientes (e.g. gradientes respecto a x_penalty)
            # eps = torch.rand(batch_size, 1).to(DEVICE)  # x shape: (batch_size, K*F)
            eps = torch.rand(batch_size, 1, 1).to(DEVICE)  # x shape: (batch_size, K, F)
            x_penalty = eps * x + (1 - eps) * x_fake
            # print(f'{x_penalty.shape=}')
            # x_penalty = x_penalty.view(x_penalty.size(0), -1)  # ??
            # print(f'{x_penalty.shape=}')
            with torch.backends.cudnn.flags(enabled=False):
                p_outputs = D(x_penalty, y)
            # print(f'{p_outputs.shape=}')
            # Calcular la suma de gradientes de salidas (outputs) respecto a las entradas (inputs)
            # xp_grad = autograd.grad(outputs=p_outputs, inputs=x_penalty, grad_outputs=C_labels,
            #                         create_graph=True, retain_graph=True, only_inputs=True)
            grad_outputs = torch.ones_like(p_outputs)
            # print(f'{grad_outputs.shape=}')
            xp_grad = autograd.grad(outputs=p_outputs, inputs=x_penalty, grad_outputs=grad_outputs, create_graph=True)
            # print(f'{xp_grad[0].shape=}')
            # grad_penalty = p_coeff * torch.mean(torch.pow(torch.norm(xp_grad[0], 2, 1) - 1, 2))
            grad_penalty = p_coeff * torch.mean(torch.pow(torch.norm(xp_grad[0], 2, 2) - 1, 2))  # !!! norma en dim 2
            # print(f'{grad_penalty=}')

            # Wasserstein loss
            x_outputs = D(x, y)
            z_outputs = D(x_fake, y)
            D_x_loss = torch.mean(x_outputs)
            D_z_loss = torch.mean(z_outputs)
            D_loss = D_z_loss - D_x_loss + grad_penalty

            D_loss.backward()
            D_opt.step()
            if dynamic_LR:
                D_scheduler.step()

            if step % n_critic == 0:
                D.zero_grad()
                G.zero_grad()
                # Entrenar el generador
                # z = torch.randn(batch_size, K).to(DEVICE)
                z = torch.randn(batch_size, K, 1).to(DEVICE)  # Z_1
                # z = torch.randn(batch_size, K, F).to(DEVICE)  # Z_n
                z_outputs = D(G(z, y).reshape(x.shape), y)
                G_loss = -torch.mean(z_outputs)

                G_loss.backward()
                G_opt.step()
                if dynamic_LR:
                    G_scheduler.step()

            step += 1

        if epoch % 1 == 0:
            # print(f'Epoch: {epoch}/{epochs} D_Loss: {D_loss.item():.4f} D_z_Loss: {D_z_loss:.4f} D_x_loss: {D_x_loss:.4f} Grad Penalty: {grad_penalty:.4f} G_Loss: {G_loss.item():.4f} Time: {(time.time() - start_time)/60:.1f}m')
            print('Epoch: ' + str(epoch/epochs) + ' D_Loss: ' + str(D_loss.item()) + 'D_z_Loss: ' + str(D_z_loss) + ' D_x_loss: ' + str(D_x_loss) + ' Grad Penalty: ' + str(grad_penalty) + ' G_Loss: ' + str(G_loss.item()) + ' Time: ' + str((time.time() - start_time)/60) + ' m')
            # print(f'Epoch: {epoch}/{epochs}, Step: {step}, D Loss: {D_loss.item()}, G Loss: {G_loss.item()}, Time: {time.time() - start_time}')
            # print('Epoch: {}/{}, Step: {}, C Loss: {}, G Loss: {}'.format(epoch, epochs, step, D_loss.item(), G_loss.item()))

            # if step % 1000 == 0:
            #     G.eval()
            #     img = get_sample_image(G, n_noise)
            #     imsave('samples/{}_step{:05d}.jpg'.format(MODEL_NAME, step), img, cmap='gray')
            #     G.train()
        # ===================================================================================

    torch.save(G.state_dict(), 'model/generator_CR_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '.pth')
    torch.save(D.state_dict(), 'model/discriminator_CR_' + str(K) + '_' + str(S) + '_' + str(epochs) + '_' + str(generator_learning_rate) + '_' + str(dynamic_LR) + '.pth')

    # generate 2019 data
    G.eval()
    D.eval()
    # original_train_data = original_data[0:3287]
    original_test_data = original_data[3287:]
    real_2019 = original_test_data['Demanda']
    # predicted_2019 = generate_mh_samples(D, G, original_test_data[:, :-1], original_train_data.sample)[:, -1]

    # Z_n
    real_2019_data = torch.zeros(test_data.shape[0] - K + 1, K, F).to(DEVICE)
    for i in range(real_2019.shape[0] - K + 1):
        real_2019_data[i, :, :] = test_data[i:i + K, :]

    y = real_2019_data[:, :, :-1]
    z = torch.randn(test_data.shape[0] - K + 1, K, 1).to(DEVICE)  # Z_1
    pred = G(z, y)


    # pred = G(real_2019_data)
    # print(pred[-1,-1])
    # print(pred.reshape(-1,K,F).shape)
    # print(pred.reshape(-1,K,F)[-1,-1,-1])
    predicted_2019 = torch.zeros(real_2019.shape[0])
    # for j in range(pred.reshape(-1, K, F).shape[0]):
    for j in range(pred.reshape(-1, K, 1).shape[0]):
        # if j != pred.reshape(-1, K, F).shape[0] - 1:
        if j != pred.reshape(-1, K, 1).shape[0] - 1:
            # if j == 0:
                # print(pred.reshape(-1,K,F)[j,:,-1])
            # predicted_2019[j] = pred.reshape(-1, K, F)[j, 0, -1]
            predicted_2019[j] = pred.reshape(-1, K, 1)[j, 0, -1]
        else:
            # print(f'{j=}')
            # print(f'{j+K=}')
            # print(pred.reshape(-1,K,F)[j,:,-1])
            # print(f'{predicted_2019[-1]=}')
            # print(f'{predicted_2019[364]=}')
            # print(f'{predicted_2019[-2]=}')
            predicted_2019[j:j + K] = pred.reshape(-1, K, 1)[j, :, -1]
            # predicted_2019[j:j + K] = pred.reshape(-1, K, F)[j, :, -1]
            # print(f'{predicted_2019[-1]=}')
            # print(f'{predicted_2019[364]=}')
            # print(f'{predicted_2019[-2]=}')
            # print(f'{predicted_2019[j:j+K].shape=}')
    # print(f'{predicted_2019.shape=}')
    print(predicted_2019)

    # DataFrame(predicted_2019.detach().numpy()).to_csv('C:/Users/D255728/Documents/ProyectoGAN/datos/predicted_2019.csv', header=False)
    rmse = sqrt(mean_squared_error(real_2019, predicted_2019.detach()))
    print('rmse=' + str(rmse))
    print('DEVICE=' + str(DEVICE))

    # # Z_1
    # gen_data = {}
    # z_i = torch.randn(100_000, K, 1)
    # x_pred = G(z_i).reshape(-1, K, F)  # (100_000, K, F)
    # # print(f'{x_pred.shape=}')

    # for i in range(x_pred.shape[0]):
    #     # for j in test_data['t']:
    #     for j in range(real_2019.shape[0] - K + 1):
    #         # if j == 0:
    #             # print(f'{x_pred[i, :, :][:-1].shape=}')
    #             # print(f'{test_data[j:j + K, :][:-1].shape=}')
    #         # print(f'{x_pred[i, :, :].shape=}')
    #         # print(f'{test_data[j:j + K, :].shape=}')
    #         # error = sum((x_pred[i, :, :][:-1] - test_data[j:j + K, :][:-1])**2)
    #         error = torch.sum((x_pred[i, :, :] - test_data[j:j + K, :])**2)
    #         # print(f'{error=}')
    #         if j in gen_data:
    #             if error < gen_data[j][0]:
    #                 # gen_data[j] = error, x_pred[i, :, :][-1]
    #                 gen_data[j] = error, x_pred[i, :, :]
    #         else:
    #             # gen_data[j] = error, x_pred[i, :, :][-1]
    #             gen_data[j] = error, x_pred[i, :, :]

    # best_predicted_2019 = {}
    # for i in range(real_2019.shape[0] - K + 1):
    #     if i == real_2019.shape[0] - K:
    #         for j in range(real_2019.shape[0] - K, real_2019.shape[0]):
    #             best_predicted_2019[j] = gen_data[i][1][i - j, -1].item()
    #     else:
    #         error = 1_000_000
    #         best_j = -1
    #         window_pos = -1
    #         for j in range(max(0, i - K + 1), i + 1):
    #             if gen_data[j][0] < error:
    #                 error = gen_data[j][0]
    #                 best_j = j
    #                 window_pos = i - j
    #                 # if window_pos == 20:
    #                 #     print(f'{i=}')
    #                 #     print(f'{j=}')
    #         best_predicted_2019[i] = gen_data[best_j][1][window_pos, -1].item()


    # predicted_2019 = torch.zeros(real_2019.shape[0])
    # for j in range(x_pred.reshape(-1, K, F).shape[0]):
    #     if j != pred.reshape(-1, K, F).shape[0] - 1:
    #         # if j == 0:
    #             # print(pred.reshape(-1,K,F)[j,:,-1])
    #         predicted_2019[j] = pred.reshape(-1, K, F)[j, 0, -1]
    #     else:
    #         # print(f'{j=}')
    #         # print(f'{j+K=}')
    #         # print(pred.reshape(-1,K,F)[j,:,-1])
    #         # print(f'{predicted_2019[-1]=}')
    #         # print(f'{predicted_2019[364]=}')
    #         # print(f'{predicted_2019[-2]=}')
    #         predicted_2019[j:j + K] = pred.reshape(-1, K, F)[j, :, -1]
    #         # print(f'{predicted_2019[-1]=}')
    #         # print(f'{predicted_2019[364]=}')
    #         # print(f'{predicted_2019[-2]=}')
    #         # print(f'{predicted_2019[j:j+K].shape=}')
    # # print(f'{predicted_2019.shape=}')
    # print(predicted_2019)

    # print(f'{gen_data=}')
    # # predicted_2019 = [x for _, x in gen_data.values()]
    # print(f'{best_predicted_2019}')
    # DataFrame(np.array([x for x in best_predicted_2019.items()])).to_csv('C:/Users/D255728/Documents/ProyectoGAN/datos/predicted_2019.csv', header=False)
    # rmse = sqrt(mean_squared_error(real_2019, list(best_predicted_2019.values())))
    # print(f'{rmse=}')


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

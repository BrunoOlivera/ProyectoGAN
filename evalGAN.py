from R_GAN_H import generate_mh_samples, Discriminator as DZ_1, Generator as GenZ_1
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from pandas import read_csv
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import sys


def eval(num_mh_samples):
    c = 128  # cell state size

    D_R_H_24_24_6000_2e_06_False_5 = DZ_1(window_size=24, hidden_size=c, feature_size=7)
    D_R_H_24_24_6000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_discriminator_R_H_24_24_6000_2e-06_False_5.pth'))
    D_R_H_24_24_6000_2e_06_False_5.eval()
    G_R_H_24_24_6000_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
    G_R_H_24_24_6000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_24_6000_2e-06_False_5.pth'))
    G_R_H_24_24_6000_2e_06_False_5.eval()
    G_R_H_24_24_6000_2e_06_False_5_500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
    G_R_H_24_24_6000_2e_06_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_24_6000_2e-06_False_5_500.pth'))
    G_R_H_24_24_6000_2e_06_False_5_500.eval()
    G_R_H_24_24_6000_2e_06_False_5_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
    G_R_H_24_24_6000_2e_06_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_24_6000_2e-06_False_5_1000.pth'))
    G_R_H_24_24_6000_2e_06_False_5_1000.eval()
    G_R_H_24_24_6000_2e_06_False_5_1500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
    G_R_H_24_24_6000_2e_06_False_5_1500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_24_6000_2e-06_False_5_1500.pth'))
    G_R_H_24_24_6000_2e_06_False_5_1500.eval()
    G_R_H_24_24_6000_2e_06_False_5_2000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
    G_R_H_24_24_6000_2e_06_False_5_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_24_6000_2e-06_False_5_2000.pth'))
    G_R_H_24_24_6000_2e_06_False_5_2000.eval()
    G_R_H_24_24_6000_2e_06_False_5_2500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
    G_R_H_24_24_6000_2e_06_False_5_2500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_24_6000_2e-06_False_5_2500.pth'))
    G_R_H_24_24_6000_2e_06_False_5_2500.eval()
    G_R_H_24_24_6000_2e_06_False_5_3000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
    G_R_H_24_24_6000_2e_06_False_5_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_24_6000_2e-06_False_5_3000.pth'))
    G_R_H_24_24_6000_2e_06_False_5_3000.eval()
    G_R_H_24_24_6000_2e_06_False_5_3500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
    G_R_H_24_24_6000_2e_06_False_5_3500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_24_6000_2e-06_False_5_3500.pth'))
    G_R_H_24_24_6000_2e_06_False_5_3500.eval()
    G_R_H_24_24_6000_2e_06_False_5_4000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
    G_R_H_24_24_6000_2e_06_False_5_4000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_24_6000_2e-06_False_5_4000.pth'))
    G_R_H_24_24_6000_2e_06_False_5_4000.eval()
    G_R_H_24_24_6000_2e_06_False_5_4500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
    G_R_H_24_24_6000_2e_06_False_5_4500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_24_6000_2e-06_False_5_4500.pth'))
    G_R_H_24_24_6000_2e_06_False_5_4500.eval()
    G_R_H_24_24_6000_2e_06_False_5_5000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
    G_R_H_24_24_6000_2e_06_False_5_5000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_24_6000_2e-06_False_5_5000.pth'))
    G_R_H_24_24_6000_2e_06_False_5_5000.eval()
    G_R_H_24_24_6000_2e_06_False_5_5500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
    G_R_H_24_24_6000_2e_06_False_5_5500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_24_6000_2e-06_False_5_5500.pth'))
    G_R_H_24_24_6000_2e_06_False_5_5500.eval()

    # Get DATA
    data = read_csv('C:/Users/D255728/Documents/ProyectoGAN/datos/datos_horarios_GAN.csv', header=0, index_col=0, sep=';', decimal=',', encoding='latin-1')

    # # categorical to numeric
    # data['tipo_dia'] = data['tipo_dia'].astype('category')
    # data['tipo_dia'] = data['tipo_dia'].cat.codes

    data = data[['Mes', 'Dia', 'Hora', 'semana', 'Temperatura', 'arima', 'Demanda']]

    # normalize data
    data[data.columns] = MinMaxScaler().fit_transform(data[data.columns])

    # split train/test
    split_idx = 78888
    train_data = data[0:split_idx]
    test_data = data[split_idx:]

    # seed = 0
    # torch.manual_seed(seed)
    # random.seed(seed)

    K = 24
    F = 7
    window = 24
    step = 24
    # step = 12

    X = test_data.values.reshape(-1, K, F)
    X = torch.tensor(X)

    # X = np.zeros((test_data.shape[0] // step, K, F))
    # # for i in range(0, test_data.shape[0] - 12, step):
    # for i in range(0, test_data.shape[0], step):
    #     X[i, :, :] = test_data.values[i:i + window, :]

    G = G_R_H_24_24_6000_2e_06_False_5
    D = D_R_H_24_24_6000_2e_06_False_5

    z_i = torch.randn(X.shape[0], K, 1)
    X_fake = G(z_i).reshape(-1, K, F)
    D.double().calibrate_discriminator(X, X_fake)

    # num_mh_samples = 3500
    # num_mh_samples = 10_000

    init_sample_idx = random.randint(0, train_data.shape[0] - 1)
    init_sample = torch.tensor(train_data.values[init_sample_idx:init_sample_idx + window, :])
    x_pred = generate_mh_samples(G, D, num_mh_samples, init_sample)

    real_data = []
    for testIdx in range(0, train_data.shape[0], step):
        plt.plot(train_data.values[testIdx:testIdx + window, -1], color='r', linestyle='-')
        real_data.append(train_data.values[testIdx:testIdx + window, -1])
    np_real_data = np.array(real_data)

    fake_data = []
    for j in range(num_mh_samples):
        plt.plot(x_pred[j, :, -1], color='b', linestyle='-')
        fake_data.append(x_pred[j, :, -1])
    np_fake_data = np.array(fake_data)

    # Coverage Metric
    kde = KernelDensity().fit(np_fake_data)

    model_log_density = kde.score_samples(np_fake_data)
    threshold = np.percentile(model_log_density, 5)
    real_points_log_density = kde.score_samples(np_real_data)
    ratio_not_covered = np.mean(real_points_log_density <= threshold)
    C = 1 - ratio_not_covered

    print(f'Coverage: {C * 100:.1f}%')

    plt.show()


if __name__ == '__main__':
    eval(int(sys.argv[1]))

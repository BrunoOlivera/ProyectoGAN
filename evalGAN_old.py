from CR_GAN import Generator as GenCR
from R_GAN_H import generate_mh_samples, Discriminator as DZ_1, Generator as GenZ_1
from RN_GAN import Generator as GenZ_N
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from pandas import read_csv, DataFrame
import torch
import matplotlib.pyplot as plt
import collections
import numpy as np
import random
import time
from collections import Counter

# # Get DATA
# data = read_csv('C:/Users/D255728/Documents/ProyectoGAN/datos/datos_diarios_GAN.csv', header=0, index_col=0, encoding='latin-1')

# # categorical to numeric
# data['E4'] = data['E4'].astype('category')
# data['E4'] = data['E4'].cat.codes
# data['tipo_dia'] = data['tipo_dia'].astype('category')
# data['tipo_dia'] = data['tipo_dia'].cat.codes

# # normalize data
# data[data.columns] = MinMaxScaler().fit_transform(data[data.columns])
# original_data = data.copy()
# data = torch.tensor(data.values)

# # split train/test
# train_data = data[0:3287]
# test_data = data[3287:]

# K = 20  # window size
# S = 1  # step size
# F = train_data.shape[-1]  # features
# # F = 1
c = 128  # cell state size

# G_CR_1500 = Generator(window_size=K, hidden_size=c, feature_size=F)
# G_CR_1500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_CR_1500.pth'))
# G_CR_1500.eval()
# G_CR_3000 = Generator(window_size=K, hidden_size=c, feature_size=F)
# G_CR_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_CR_3000.pth'))
# G_CR_3000.eval()
# G_CR_6000 = Generator(window_size=K, hidden_size=c, feature_size=F)
# G_CR_6000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_CR_6000.pth'))
# G_CR_6000.eval()
# G_CR_10_5_6000 = Generator(window_size=10, hidden_size=c, feature_size=F)
# G_CR_10_5_6000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_CR_10_5_6000.pth'))
# G_CR_10_5_6000.eval()
# G_CR_10_2_6000 = Generator(window_size=10, hidden_size=c, feature_size=F)
# G_CR_10_2_6000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_10_2.pth'))
# G_CR_10_2_6000.eval()
# G_CR_10_1_6000 = Generator(window_size=10, hidden_size=c, feature_size=F)
# G_CR_10_1_6000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_CR_10_1_6000_True.pth'))
# G_CR_10_1_6000.eval()
# G_CR_20_1_3000 = Generator(window_size=K, hidden_size=c, feature_size=F)
# G_CR_20_1_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_CR_20_1_3000_0.0001_False.pth'))
# G_CR_20_1_3000.eval()
# G_CR_30_1_6000 = Generator(window_size=30, hidden_size=c, feature_size=F)
# G_CR_30_1_6000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_CR_30_16000_True.pth'))
# G_CR_30_1_6000.eval()
# G_CR_20_1_6000 = Generator(window_size=K, hidden_size=c, feature_size=F)
# G_CR_20_1_6000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_CR_20_1_6000_0.0001_False.pth'))
# G_CR_20_1_6000.eval()
# G_CR_20_1_3000_2e_6 = Generator(window_size=K, hidden_size=c, feature_size=F)
# G_CR_20_1_3000_2e_6.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_CR_20_1_3000_2e-06_False.pth'))
# G_CR_20_1_3000_2e_6.eval()
# G_CR_20_1_3000_2e_5 = Generator(window_size=K, hidden_size=c, feature_size=F)
# G_CR_20_1_3000_2e_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_CR_20_1_3000_2e-05_False.pth'))
# G_CR_20_1_3000_2e_5.eval()
# # G_CR_20_1_3000_2e_5_2 = Generator(window_size=K, hidden_size=c, feature_size=F)
# # G_CR_20_1_3000_2e_5_2.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_CR_20_1_3000_2e-05_False_2.pth'))
# # G_CR_20_1_3000_2e_5_2.eval()
# G_CR_60_30_1500_2e_6_False_5 = Generator(window_size=60, hidden_size=c, feature_size=12)
# G_CR_60_30_1500_2e_6_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_CR_H_60_30_1500_2e-06_False_5.pth'))
# G_CR_60_30_1500_2e_6_False_5.eval()
# G_CR_24_24_1500_2e_6_False_5 = Generator(window_size=24, hidden_size=c, feature_size=12)
# G_CR_24_24_1500_2e_6_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_CR_H_24_24_1500_2e-06_False_5.pth'))
# G_CR_24_24_1500_2e_6_False_5.eval()
#
# G_CR_24_12_6000_2e_6_False_5 = GenCR(window_size=24, hidden_size=c, feature_size=7)
# G_CR_24_12_6000_2e_6_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_CR_H_24_12_6000_2e-06_False_5.pth'))
# G_CR_24_12_6000_2e_6_False_5.eval()
# G_CR_24_12_6000_2e_6_False_5_500 = GenCR(window_size=24, hidden_size=c, feature_size=7)
# G_CR_24_12_6000_2e_6_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_24_12_6000_2e-06_False_5_500.pth'))
# G_CR_24_12_6000_2e_6_False_5_500.eval()
# G_CR_24_12_6000_2e_6_False_5_1000 = GenCR(window_size=24, hidden_size=c, feature_size=7)
# G_CR_24_12_6000_2e_6_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_24_12_6000_2e-06_False_5_1000.pth'))
# G_CR_24_12_6000_2e_6_False_5_1000.eval()
# G_CR_24_12_6000_2e_6_False_5_1500 = GenCR(window_size=24, hidden_size=c, feature_size=7)
# G_CR_24_12_6000_2e_6_False_5_1500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_24_12_6000_2e-06_False_5_1500.pth'))
# G_CR_24_12_6000_2e_6_False_5_1500.eval()
# G_CR_24_12_6000_2e_6_False_5_2000 = GenCR(window_size=24, hidden_size=c, feature_size=7)
# G_CR_24_12_6000_2e_6_False_5_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_24_12_6000_2e-06_False_5_2000.pth'))
# G_CR_24_12_6000_2e_6_False_5_2000.eval()
# G_CR_24_12_6000_2e_6_False_5_2500 = GenCR(window_size=24, hidden_size=c, feature_size=7)
# G_CR_24_12_6000_2e_6_False_5_2500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_24_12_6000_2e-06_False_5_2500.pth'))
# G_CR_24_12_6000_2e_6_False_5_2500.eval()
# G_CR_24_12_6000_2e_6_False_5_3000 = GenCR(window_size=24, hidden_size=c, feature_size=7)
# G_CR_24_12_6000_2e_6_False_5_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_24_12_6000_2e-06_False_5_3000.pth'))
# G_CR_24_12_6000_2e_6_False_5_3000.eval()
# G_CR_24_12_6000_2e_6_False_5_3500 = GenCR(window_size=24, hidden_size=c, feature_size=7)
# G_CR_24_12_6000_2e_6_False_5_3500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_24_12_6000_2e-06_False_5_3500.pth'))
# G_CR_24_12_6000_2e_6_False_5_3500.eval()
# G_CR_24_12_6000_2e_6_False_5_4000 = GenCR(window_size=24, hidden_size=c, feature_size=7)
# G_CR_24_12_6000_2e_6_False_5_4000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_24_12_6000_2e-06_False_5_4000.pth'))
# G_CR_24_12_6000_2e_6_False_5_4000.eval()
# G_CR_24_12_6000_2e_6_False_5_4500 = GenCR(window_size=24, hidden_size=c, feature_size=7)
# G_CR_24_12_6000_2e_6_False_5_4500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_24_12_6000_2e-06_False_5_4500.pth'))
# G_CR_24_12_6000_2e_6_False_5_4500.eval()
# G_CR_24_12_6000_2e_6_False_5_5000 = GenCR(window_size=24, hidden_size=c, feature_size=7)
# G_CR_24_12_6000_2e_6_False_5_5000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_24_12_6000_2e-06_False_5_5000.pth'))
# G_CR_24_12_6000_2e_6_False_5_5000.eval()
# G_CR_24_12_6000_2e_6_False_5_5500 = GenCR(window_size=24, hidden_size=c, feature_size=7)
# G_CR_24_12_6000_2e_6_False_5_5500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_24_12_6000_2e-06_False_5_5500.pth'))
# G_CR_24_12_6000_2e_6_False_5_5500.eval()
# G_CR_60_30_6000_2e_6_False_5 = GenCR(window_size=60, hidden_size=c, feature_size=7)
# G_CR_60_30_6000_2e_6_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_CR_H_60_30_6000_2e-06_False_5.pth'))
# G_CR_60_30_6000_2e_6_False_5.eval()
# G_CR_60_30_6000_2e_6_False_5_500 = GenCR(window_size=60, hidden_size=c, feature_size=7)
# G_CR_60_30_6000_2e_6_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_60_30_6000_2e-06_False_5_500.pth'))
# G_CR_60_30_6000_2e_6_False_5_500.eval()
# G_CR_60_30_6000_2e_6_False_5_1000 = GenCR(window_size=60, hidden_size=c, feature_size=7)
# G_CR_60_30_6000_2e_6_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_60_30_6000_2e-06_False_5_1000.pth'))
# G_CR_60_30_6000_2e_6_False_5_1000.eval()
# G_CR_60_30_6000_2e_6_False_5_1500 = GenCR(window_size=60, hidden_size=c, feature_size=7)
# G_CR_60_30_6000_2e_6_False_5_1500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_60_30_6000_2e-06_False_5_1500.pth'))
# G_CR_60_30_6000_2e_6_False_5_1500.eval()
# G_CR_60_30_6000_2e_6_False_5_2000 = GenCR(window_size=60, hidden_size=c, feature_size=7)
# G_CR_60_30_6000_2e_6_False_5_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_60_30_6000_2e-06_False_5_2000.pth'))
# G_CR_60_30_6000_2e_6_False_5_2000.eval()
# G_CR_60_30_6000_2e_6_False_5_2500 = GenCR(window_size=60, hidden_size=c, feature_size=7)
# G_CR_60_30_6000_2e_6_False_5_2500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_60_30_6000_2e-06_False_5_2500.pth'))
# G_CR_60_30_6000_2e_6_False_5_2500.eval()
# G_CR_60_30_6000_2e_6_False_5_3000 = GenCR(window_size=60, hidden_size=c, feature_size=7)
# G_CR_60_30_6000_2e_6_False_5_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_60_30_6000_2e-06_False_5_3000.pth'))
# G_CR_60_30_6000_2e_6_False_5_3000.eval()
# G_CR_60_30_6000_2e_6_False_5_3500 = GenCR(window_size=60, hidden_size=c, feature_size=7)
# G_CR_60_30_6000_2e_6_False_5_3500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_60_30_6000_2e-06_False_5_3500.pth'))
# G_CR_60_30_6000_2e_6_False_5_3500.eval()
# G_CR_60_30_6000_2e_6_False_5_4000 = GenCR(window_size=60, hidden_size=c, feature_size=7)
# G_CR_60_30_6000_2e_6_False_5_4000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_60_30_6000_2e-06_False_5_4000.pth'))
# G_CR_60_30_6000_2e_6_False_5_4000.eval()
# G_CR_60_30_6000_2e_6_False_5_4500 = GenCR(window_size=60, hidden_size=c, feature_size=7)
# G_CR_60_30_6000_2e_6_False_5_4500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_60_30_6000_2e-06_False_5_4500.pth'))
# G_CR_60_30_6000_2e_6_False_5_4500.eval()
# G_CR_60_30_6000_2e_6_False_5_5000 = GenCR(window_size=60, hidden_size=c, feature_size=7)
# G_CR_60_30_6000_2e_6_False_5_5000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_60_30_6000_2e-06_False_5_5000.pth'))
# G_CR_60_30_6000_2e_6_False_5_5000.eval()
# G_CR_60_30_6000_2e_6_False_5_5500 = GenCR(window_size=60, hidden_size=c, feature_size=7)
# G_CR_60_30_6000_2e_6_False_5_5500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/CR_H/generator_R_H_60_30_6000_2e-06_False_5_5500.pth'))
# G_CR_60_30_6000_2e_6_False_5_5500.eval()

#

# # Z_1
# G_CR_____ = GenZ_1(window_size=K, hidden_size=c, feature_size=F)
# G_CR_____.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_(_).pth'))
# G_CR_____.eval()
# G_Z1 = GenZ_1(window_size=K, hidden_size=c, feature_size=F)
# G_Z1.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_Z1.pth'))
# G_Z1.eval()
# G_R_20_1_3000_2e_5 = GenZ_1(window_size=K, hidden_size=c, feature_size=F)
# G_R_20_1_3000_2e_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_R_20_1_3000_2e-05_False.pth'))
# G_R_20_1_3000_2e_5.eval()
# G_R_20_1_3000_2e_6 = GenZ_1(window_size=K, hidden_size=c, feature_size=F)
# G_R_20_1_3000_2e_6.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_R_20_1_3000_2e-06_False.pth'))
# G_R_20_1_3000_2e_6.eval()
# G_R_20_1_1500_2e_6 = GenZ_1(window_size=K, hidden_size=c, feature_size=F)
# G_R_20_1_1500_2e_6.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_R_20_1_1500_2e-06_False.pth'))
# G_R_20_1_1500_2e_6.eval()
# G_R_20_1_1500_2e_5 = GenZ_1(window_size=K, hidden_size=c, feature_size=F)
# G_R_20_1_1500_2e_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_R_20_1_1500_2e-05_False.pth'))
# G_R_20_1_1500_2e_5.eval()
# G_R_20_1_3000_2e_06_False_5_best_1000 = GenZ_1(window_size=K, hidden_size=c, feature_size=F)
# G_R_20_1_3000_2e_06_False_5_best_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R/generator_R_20_1_3000_2e-06_False_5_best_1000.pth'))
# G_R_20_1_3000_2e_06_False_5_best_1000.eval()
# G_R_20_1_3000_2e_06_False_5_best_2000 = GenZ_1(window_size=K, hidden_size=c, feature_size=F)
# G_R_20_1_3000_2e_06_False_5_best_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R/generator_R_20_1_3000_2e-06_False_5_best_2000.pth'))
# G_R_20_1_3000_2e_06_False_5_best_2000.eval()
# G_R_20_1_3000_2e_06_False_5_best_3000 = GenZ_1(window_size=K, hidden_size=c, feature_size=F)
# G_R_20_1_3000_2e_06_False_5_best_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R/generator_R_20_1_3000_2e-06_False_5_best_3000.pth'))
# G_R_20_1_3000_2e_06_False_5_best_3000.eval()
# G_R_20_1_3000_2e_06_False_5 = GenZ_1(window_size=K, hidden_size=c, feature_size=F)
# G_R_20_1_3000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R/generator_R_20_1_3000_2e-06_False_5.pth'))
# G_R_20_1_3000_2e_06_False_5.eval()

# # R_H
# G_R_H_60_30_1500_2e_06_False_5 = GenZ_1(window_size=60, hidden_size=c, feature_size=12)
# G_R_H_60_30_1500_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_60_30_1500_2e-06_False_5.pth'))
# G_R_H_60_30_1500_2e_06_False_5.eval()
# G_R_H_24_5_3000_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_5_3000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_24_5_3000_2e-06_False_5.pth'))
# G_R_H_24_5_3000_2e_06_False_5.eval()
# G_R_H_60_30_3000_2e_06_False_5 = GenZ_1(window_size=60, hidden_size=c, feature_size=12)
# G_R_H_60_30_3000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_3000_2e-06_False_5.pth'))
# G_R_H_60_30_3000_2e_06_False_5.eval()
# G_R_H_24_5_10000_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_5_10000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_5_10000_2e-06_False_5.pth'))
# G_R_H_24_5_10000_2e_06_False_5.eval()
# G_R_H_60_30_10000_2e_06_False_5 = GenZ_1(window_size=60, hidden_size=c, feature_size=12)
# G_R_H_60_30_10000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_10000_2e-06_False_5.pth'))
# G_R_H_60_30_10000_2e_06_False_5.eval()
# G_R_H_60_30_1500_2e_06_False_5 = GenZ_1(window_size=60, hidden_size=c, feature_size=12)
# G_R_H_60_30_1500_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_1500_2e-06_False_5.pth'))
# G_R_H_60_30_1500_2e_06_False_5.eval()
# G_R_H_60_30_1500_2e_06_False_5_best_1000 = GenZ_1(window_size=60, hidden_size=c, feature_size=12)
# G_R_H_60_30_1500_2e_06_False_5_best_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_1500_2e-06_False_5_best_1000.pth'))
# G_R_H_60_30_1500_2e_06_False_5_best_1000.eval()
# G_R_H_60_30_1500_2e_06_False_5_best_2000 = GenZ_1(window_size=60, hidden_size=c, feature_size=12)
# G_R_H_60_30_1500_2e_06_False_5_best_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_1500_2e-06_False_5_best_2000.pth'))
# G_R_H_60_30_1500_2e_06_False_5_best_2000.eval()
# # G_R_H_60_30_1500_2e_06_False_5_best_3000 = GenZ_1(window_size=60, hidden_size=c, feature_size=12)
# # G_R_H_60_30_1500_2e_06_False_5_best_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_1500_2e-06_False_5_best_3000.pth'))
# # G_R_H_60_30_1500_2e_06_False_5_best_3000.eval()
# G_R_H_24_7_1500_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_7_1500_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_7_1500_2e-06_False_5.pth'))
# G_R_H_24_7_1500_2e_06_False_5.eval()
# G_R_H_24_7_1500_2e_06_False_5_best_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_7_1500_2e_06_False_5_best_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_7_1500_2e-06_False_5_best_1000.pth'))
# G_R_H_24_7_1500_2e_06_False_5_best_1000.eval()
# G_R_H_24_7_1500_2e_06_False_5_best_2000 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_7_1500_2e_06_False_5_best_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_7_1500_2e-06_False_5_best_2000.pth'))
# G_R_H_24_7_1500_2e_06_False_5_best_2000.eval()
# G_R_H_60_30_1500_2e_06_False_5_best_1000_1_layer = GenZ_1(window_size=60, hidden_size=c, feature_size=12)
# G_R_H_60_30_1500_2e_06_False_5_best_1000_1_layer.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/1_layer_generator_R_H_60_30_1500_2e-06_False_5_best_1000.pth'), strict=False)
# G_R_H_60_30_1500_2e_06_False_5_best_1000_1_layer.eval()
# G_R_H_24_12_1500_2e_06_False_5_best_1000_1_layer = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_12_1500_2e_06_False_5_best_1000_1_layer.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/1_layer_generator_R_H_24_12_1500_2e-06_False_5_best_1000.pth'), strict=False)
# G_R_H_24_12_1500_2e_06_False_5_best_1000_1_layer.eval()
# G_R_H_24_12_1500_2e_06_False_5_1_layer = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_12_1500_2e_06_False_5_1_layer.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/1_layer_generator_R_H_24_12_1500_2e-06_False_5.pth'), strict=False)
# G_R_H_24_12_1500_2e_06_False_5_1_layer.eval()
# G_R_H_24_24_3000_2e_06_False_2 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_24_3000_2e_06_False_2.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_24_3000_2e-06_False_2.pth'))
# G_R_H_24_24_3000_2e_06_False_2.eval()
# G_R_H_24_24_3000_2e_06_False_2_best_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_24_3000_2e_06_False_2_best_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_24_3000_2e-06_False_2_best_1000.pth'))
# G_R_H_24_24_3000_2e_06_False_2_best_1000.eval()
# G_R_H_24_24_3000_2e_06_False_2_best_2000 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_24_3000_2e_06_False_2_best_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_24_3000_2e-06_False_2_best_2000.pth'))
# G_R_H_24_24_3000_2e_06_False_2_best_2000.eval()
# G_R_H_24_24_3000_2e_06_False_2_best_3000 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_24_3000_2e_06_False_2_best_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_24_3000_2e-06_False_2_best_3000.pth'))
# G_R_H_24_24_3000_2e_06_False_2_best_3000.eval()
# G_R_H_24_24_1500_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_24_1500_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_24_1500_2e-06_False_5.pth'))
# G_R_H_24_24_1500_2e_06_False_5.eval()
# G_R_H_24_24_1500_2e_06_False_5_best_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_24_1500_2e_06_False_5_best_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_24_1500_2e-06_False_5_best_1000.pth'))
# G_R_H_24_24_1500_2e_06_False_5_best_1000.eval()
# G_R_H_24_24_1500_2e_06_False_5_best_2000 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_24_1500_2e_06_False_5_best_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_24_1500_2e-06_False_5_best_2000.pth'))
# G_R_H_24_24_1500_2e_06_False_5_best_2000.eval()
# # G_R_H_24_24_3000_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# # G_R_H_24_24_3000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_24_3000_2e-06_False_5.pth'))
# # G_R_H_24_24_3000_2e_06_False_5.eval()
# G_R_H_24_24_3000_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_24_3000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_24_3000_2e-06_False_5.pth'))
# G_R_H_24_24_3000_2e_06_False_5.eval()
# G_R_H_24_24_3000_2e_06_False_5_best_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_24_3000_2e_06_False_5_best_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_24_3000_2e-06_False_5_best_1000.pth'))
# G_R_H_24_24_3000_2e_06_False_5_best_1000.eval()
# G_R_H_24_24_3000_2e_06_False_5_best_2000 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_24_3000_2e_06_False_5_best_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_24_3000_2e-06_False_5_best_2000.pth'))
# G_R_H_24_24_3000_2e_06_False_5_best_2000.eval()
# G_R_H_24_24_3000_2e_06_False_5_best_3000 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# G_R_H_24_24_3000_2e_06_False_5_best_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_24_3000_2e-06_False_5_best_3000.pth'))
# G_R_H_24_24_3000_2e_06_False_5_best_3000.eval()
# # G_R_H_24_12_1500_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# # G_R_H_24_12_1500_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_1500_2e-06_False_5.pth'))
# # G_R_H_24_12_1500_2e_06_False_5.eval()
# # G_R_H_24_12_1500_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# # G_R_H_24_12_1500_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_1500_2e-06_False_5.pth'))
# # G_R_H_24_12_1500_2e_06_False_5.eval()
# G_R_H_24_12_1500_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=1)
# G_R_H_24_12_1500_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_1500_2e-06_False_5.pth'))
# G_R_H_24_12_1500_2e_06_False_5.eval()
# # G_R_H_24_12_1500_2e_06_False_5_best_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# # G_R_H_24_12_1500_2e_06_False_5_best_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_1500_2e-06_False_5_best_1000.pth'))
# # G_R_H_24_12_1500_2e_06_False_5_best_1000.eval()
# # G_R_H_24_12_1500_2e_06_False_5_best_2000 = GenZ_1(window_size=24, hidden_size=c, feature_size=12)
# # G_R_H_24_12_1500_2e_06_False_5_best_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_1500_2e-06_False_5_best_2000.pth'))
# # G_R_H_24_12_1500_2e_06_False_5_best_2000.eval()
# # G_R_H_24_12_1500_2e_06_False_5_500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# # G_R_H_24_12_1500_2e_06_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_1500_2e-06_False_5_500.pth'))
# # G_R_H_24_12_1500_2e_06_False_5_500.eval()
# # G_R_H_24_12_1500_2e_06_False_5_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# # G_R_H_24_12_1500_2e_06_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_1500_2e-06_False_5_1000.pth'))
# # G_R_H_24_12_1500_2e_06_False_5_1000.eval()

# G_R_H_24_12_3000_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_3000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/backup/generator_R_H_24_12_3000_2e-06_False_5.pth'))
# G_R_H_24_12_3000_2e_06_False_5.eval()
# G_R_H_24_12_3000_2e_06_False_5_500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_3000_2e_06_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_3000_2e-06_False_5_500.pth'))
# G_R_H_24_12_3000_2e_06_False_5_500.eval()
# G_R_H_24_12_3000_2e_06_False_5_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_3000_2e_06_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_3000_2e-06_False_5_1000.pth'))
# G_R_H_24_12_3000_2e_06_False_5_1000.eval()
# G_R_H_24_12_3000_2e_06_False_5_1500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_3000_2e_06_False_5_1500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_3000_2e-06_False_5_1500.pth'))
# G_R_H_24_12_3000_2e_06_False_5_1500.eval()
# G_R_H_24_12_3000_2e_06_False_5_2000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_3000_2e_06_False_5_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_3000_2e-06_False_5_2000.pth'))
# G_R_H_24_12_3000_2e_06_False_5_2000.eval()
# G_R_H_24_12_3000_2e_06_False_5_2500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_3000_2e_06_False_5_2500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_3000_2e-06_False_5_2500.pth'))
# G_R_H_24_12_3000_2e_06_False_5_2500.eval()
# G_R_H_24_12_6000_2e_06_False_5_500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_06_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-06_False_5_500.pth'))
# G_R_H_24_12_6000_2e_06_False_5_500.eval()
# G_R_H_24_12_6000_2e_06_False_5_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_06_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-06_False_5_1000.pth'))
# G_R_H_24_12_6000_2e_06_False_5_1000.eval()
# G_R_H_24_12_6000_2e_06_False_5_1500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_06_False_5_1500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-06_False_5_1500.pth'))
# G_R_H_24_12_6000_2e_06_False_5_1500.eval()
# G_R_H_24_12_6000_2e_06_False_5_2000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_06_False_5_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-06_False_5_2000.pth'))
# G_R_H_24_12_6000_2e_06_False_5_2000.eval()
# G_R_H_24_12_6000_2e_06_False_5_2500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_06_False_5_2500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-06_False_5_2500.pth'))
# G_R_H_24_12_6000_2e_06_False_5_2500.eval()
# G_R_H_24_12_6000_2e_06_False_5_3000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_06_False_5_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-06_False_5_3000.pth'))
# G_R_H_24_12_6000_2e_06_False_5_3000.eval()
# G_R_H_24_12_6000_2e_06_False_5_3500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_06_False_5_3500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-06_False_5_3500.pth'))
# G_R_H_24_12_6000_2e_06_False_5_3500.eval()
# G_R_H_24_12_6000_2e_06_False_5_4000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_06_False_5_4000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-06_False_5_4000.pth'))
# G_R_H_24_12_6000_2e_06_False_5_4000.eval()
# G_R_H_24_12_6000_2e_06_False_5_4500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_06_False_5_4500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-06_False_5_4500.pth'))
# G_R_H_24_12_6000_2e_06_False_5_4500.eval()
# G_R_H_24_12_6000_2e_06_False_5_5000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_06_False_5_5000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-06_False_5_5000.pth'))
# G_R_H_24_12_6000_2e_06_False_5_5000.eval()
# G_R_H_24_12_6000_2e_06_False_5_5500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_06_False_5_5500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-06_False_5_5500.pth'))
# G_R_H_24_12_6000_2e_06_False_5_5500.eval()
# G_R_H_24_12_6000_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-06_False_5.pth'))
# G_R_H_24_12_6000_2e_06_False_5.eval()
# G_R_H_24_12_6000_2e_04_False_5_500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_04_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_0.0002_False_5_500.pth'))
# G_R_H_24_12_6000_2e_04_False_5_500.eval()
# G_R_H_24_12_6000_2e_04_False_5_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_04_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_0.0002_False_5_1000.pth'))
# G_R_H_24_12_6000_2e_04_False_5_1000.eval()
# G_R_H_24_12_6000_2e_04_False_5_1500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_04_False_5_1500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_0.0002_False_5_1500.pth'))
# G_R_H_24_12_6000_2e_04_False_5_1500.eval()
# G_R_H_24_12_6000_2e_04_False_5_2000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_04_False_5_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_0.0002_False_5_2000.pth'))
# G_R_H_24_12_6000_2e_04_False_5_2000.eval()
# G_R_H_24_12_6000_2e_04_False_5_2500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_04_False_5_2500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_0.0002_False_5_2500.pth'))
# G_R_H_24_12_6000_2e_04_False_5_2500.eval()
# G_R_H_24_12_6000_2e_04_False_5_3000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_04_False_5_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_0.0002_False_5_3000.pth'))
# G_R_H_24_12_6000_2e_04_False_5_3000.eval()
# G_R_H_24_12_6000_2e_04_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_04_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_0.0002_False_5.pth'))
# G_R_H_24_12_6000_2e_04_False_5.eval()
# G_R_H_24_12_6000_2e_05_False_5_500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_05_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-05_False_5_500.pth'))
# G_R_H_24_12_6000_2e_05_False_5_500.eval()
# G_R_H_24_12_6000_2e_05_False_5_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_05_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-05_False_5_1000.pth'))
# G_R_H_24_12_6000_2e_05_False_5_1000.eval()
# G_R_H_24_12_6000_2e_05_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_05_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-05_False_5.pth'))
# G_R_H_24_12_6000_2e_05_False_5.eval()
# G_R_H_24_12_6000_2e_07_False_5_500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_07_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-07_False_5_500.pth'))
# G_R_H_24_12_6000_2e_07_False_5_500.eval()
# G_R_H_24_12_6000_2e_07_False_5_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_07_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-07_False_5_1000.pth'))
# G_R_H_24_12_6000_2e_07_False_5_1000.eval()
# G_R_H_24_12_6000_2e_07_False_5_1500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_07_False_5_1500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-07_False_5_1500.pth'))
# G_R_H_24_12_6000_2e_07_False_5_1500.eval()
# G_R_H_24_12_6000_2e_07_False_5_2000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_07_False_5_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-07_False_5_2000.pth'))
# G_R_H_24_12_6000_2e_07_False_5_2000.eval()
# G_R_H_24_12_6000_2e_07_False_5_2500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_07_False_5_2500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-07_False_5_2500.pth'))
# G_R_H_24_12_6000_2e_07_False_5_2500.eval()
# G_R_H_24_12_6000_2e_07_False_5_3000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_07_False_5_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-07_False_5_3000.pth'))
# G_R_H_24_12_6000_2e_07_False_5_3000.eval()
# G_R_H_24_12_6000_2e_07_False_5_3500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_07_False_5_3500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-07_False_5_3500.pth'))
# G_R_H_24_12_6000_2e_07_False_5_3500.eval()
# G_R_H_24_12_6000_2e_07_False_5_4000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_07_False_5_4000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-07_False_5_4000.pth'))
# G_R_H_24_12_6000_2e_07_False_5_4000.eval()
# G_R_H_24_12_6000_2e_07_False_5_4500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_07_False_5_4500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-07_False_5_4500.pth'))
# G_R_H_24_12_6000_2e_07_False_5_4500.eval()
# G_R_H_24_12_6000_2e_07_False_5_5000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_07_False_5_5000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-07_False_5_5000.pth'))
# G_R_H_24_12_6000_2e_07_False_5_5000.eval()
# G_R_H_24_12_6000_2e_07_False_5_5500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_07_False_5_5500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-07_False_5_5500.pth'))
# G_R_H_24_12_6000_2e_07_False_5_5500.eval()
# G_R_H_24_12_6000_2e_07_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_24_12_6000_2e_07_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_24_12_6000_2e-07_False_5.pth'))
# G_R_H_24_12_6000_2e_07_False_5.eval()
#
# G_R_H_GP_24_12_6000_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_500.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_500.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_1000.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_1000.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_1500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_1500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_1500.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_1500.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_2000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_2000.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_2000.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_2500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_2500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_2500.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_2500.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_3000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_3000.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_3000.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_3500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_3500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_3500.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_3500.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_4000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_4000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_4000.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_4000.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_4500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_4500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_4500.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_4500.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_5000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_5000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_5000.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_5000.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_5500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_5500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_5500.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_5500.eval()
#
D_R_H_24_12_6000_2e_06_False_5 = DZ_1(window_size=24, hidden_size=c, feature_size=7)
D_R_H_24_12_6000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/backup/SIN_GP/discriminator_R_H_24_12_6000_2e-06_False_5.pth'))
D_R_H_24_12_6000_2e_06_False_5.eval()
G_R_H_24_12_6000_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
G_R_H_24_12_6000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/backup/SIN_GP/generator_R_H_24_12_6000_2e-06_False_5.pth'))
G_R_H_24_12_6000_2e_06_False_5.eval()
G_R_H_24_12_6000_2e_06_False_5_500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
G_R_H_24_12_6000_2e_06_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/backup/SIN_GP/generator_R_H_24_12_6000_2e-06_False_5_500.pth'))
G_R_H_24_12_6000_2e_06_False_5_500.eval()
G_R_H_24_12_6000_2e_06_False_5_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
G_R_H_24_12_6000_2e_06_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/backup/SIN_GP/generator_R_H_24_12_6000_2e-06_False_5_1000.pth'))
G_R_H_24_12_6000_2e_06_False_5_1000.eval()
G_R_H_24_12_6000_2e_06_False_5_1500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
G_R_H_24_12_6000_2e_06_False_5_1500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/backup/SIN_GP/generator_R_H_24_12_6000_2e-06_False_5_1500.pth'))
G_R_H_24_12_6000_2e_06_False_5_1500.eval()
G_R_H_24_12_6000_2e_06_False_5_2000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
G_R_H_24_12_6000_2e_06_False_5_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/backup/SIN_GP/generator_R_H_24_12_6000_2e-06_False_5_2000.pth'))
G_R_H_24_12_6000_2e_06_False_5_2000.eval()
G_R_H_24_12_6000_2e_06_False_5_2500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
G_R_H_24_12_6000_2e_06_False_5_2500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/backup/SIN_GP/generator_R_H_24_12_6000_2e-06_False_5_2500.pth'))
G_R_H_24_12_6000_2e_06_False_5_2500.eval()
G_R_H_24_12_6000_2e_06_False_5_3000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
G_R_H_24_12_6000_2e_06_False_5_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/backup/SIN_GP/generator_R_H_24_12_6000_2e-06_False_5_3000.pth'))
G_R_H_24_12_6000_2e_06_False_5_3000.eval()
G_R_H_24_12_6000_2e_06_False_5_3500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
G_R_H_24_12_6000_2e_06_False_5_3500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/backup/SIN_GP/generator_R_H_24_12_6000_2e-06_False_5_3500.pth'))
G_R_H_24_12_6000_2e_06_False_5_3500.eval()
G_R_H_24_12_6000_2e_06_False_5_4000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
G_R_H_24_12_6000_2e_06_False_5_4000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/backup/SIN_GP/generator_R_H_24_12_6000_2e-06_False_5_4000.pth'))
G_R_H_24_12_6000_2e_06_False_5_4000.eval()
G_R_H_24_12_6000_2e_06_False_5_4500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
G_R_H_24_12_6000_2e_06_False_5_4500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/backup/SIN_GP/generator_R_H_24_12_6000_2e-06_False_5_4500.pth'))
G_R_H_24_12_6000_2e_06_False_5_4500.eval()
G_R_H_24_12_6000_2e_06_False_5_5000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
G_R_H_24_12_6000_2e_06_False_5_5000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/backup/SIN_GP/generator_R_H_24_12_6000_2e-06_False_5_5000.pth'))
G_R_H_24_12_6000_2e_06_False_5_5000.eval()
G_R_H_24_12_6000_2e_06_False_5_5500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
G_R_H_24_12_6000_2e_06_False_5_5500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/backup/SIN_GP/generator_R_H_24_12_6000_2e-06_False_5_5500.pth'))
G_R_H_24_12_6000_2e_06_False_5_5500.eval()
#
#
# G_R_H_60_30_6000_2e_06_False_5 = GenZ_1(window_size=60, hidden_size=c, feature_size=7)
# G_R_H_60_30_6000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_6000_2e-06_False_5.pth'))
# G_R_H_60_30_6000_2e_06_False_5.eval()
# G_R_H_60_30_6000_2e_06_False_5_500 = GenZ_1(window_size=60, hidden_size=c, feature_size=7)
# G_R_H_60_30_6000_2e_06_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_6000_2e-06_False_5_500.pth'))
# G_R_H_60_30_6000_2e_06_False_5_500.eval()
# G_R_H_60_30_6000_2e_06_False_5_1000 = GenZ_1(window_size=60, hidden_size=c, feature_size=7)
# G_R_H_60_30_6000_2e_06_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_6000_2e-06_False_5_1000.pth'))
# G_R_H_60_30_6000_2e_06_False_5_1000.eval()
# G_R_H_60_30_6000_2e_06_False_5_1500 = GenZ_1(window_size=60, hidden_size=c, feature_size=7)
# G_R_H_60_30_6000_2e_06_False_5_1500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_6000_2e-06_False_5_1500.pth'))
# G_R_H_60_30_6000_2e_06_False_5_1500.eval()
# G_R_H_60_30_6000_2e_06_False_5_2000 = GenZ_1(window_size=60, hidden_size=c, feature_size=7)
# G_R_H_60_30_6000_2e_06_False_5_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_6000_2e-06_False_5_2000.pth'))
# G_R_H_60_30_6000_2e_06_False_5_2000.eval()
# G_R_H_60_30_6000_2e_06_False_5_2500 = GenZ_1(window_size=60, hidden_size=c, feature_size=7)
# G_R_H_60_30_6000_2e_06_False_5_2500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_6000_2e-06_False_5_2500.pth'))
# G_R_H_60_30_6000_2e_06_False_5_2500.eval()
# G_R_H_60_30_6000_2e_06_False_5_3000 = GenZ_1(window_size=60, hidden_size=c, feature_size=7)
# G_R_H_60_30_6000_2e_06_False_5_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_6000_2e-06_False_5_3000.pth'))
# G_R_H_60_30_6000_2e_06_False_5_3000.eval()
# G_R_H_60_30_6000_2e_06_False_5_3500 = GenZ_1(window_size=60, hidden_size=c, feature_size=7)
# G_R_H_60_30_6000_2e_06_False_5_3500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_6000_2e-06_False_5_3500.pth'))
# G_R_H_60_30_6000_2e_06_False_5_3500.eval()
# G_R_H_60_30_6000_2e_06_False_5_4000 = GenZ_1(window_size=60, hidden_size=c, feature_size=7)
# G_R_H_60_30_6000_2e_06_False_5_4000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_6000_2e-06_False_5_4000.pth'))
# G_R_H_60_30_6000_2e_06_False_5_4000.eval()
# G_R_H_60_30_6000_2e_06_False_5_4500 = GenZ_1(window_size=60, hidden_size=c, feature_size=7)
# G_R_H_60_30_6000_2e_06_False_5_4500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_6000_2e-06_False_5_4500.pth'))
# G_R_H_60_30_6000_2e_06_False_5_4500.eval()
# G_R_H_60_30_6000_2e_06_False_5_5000 = GenZ_1(window_size=60, hidden_size=c, feature_size=7)
# G_R_H_60_30_6000_2e_06_False_5_5000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_6000_2e-06_False_5_5000.pth'))
# G_R_H_60_30_6000_2e_06_False_5_5000.eval()
# G_R_H_60_30_6000_2e_06_False_5_5500 = GenZ_1(window_size=60, hidden_size=c, feature_size=7)
# G_R_H_60_30_6000_2e_06_False_5_5500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/generator_R_H_60_30_6000_2e-06_False_5_5500.pth'))
# G_R_H_60_30_6000_2e_06_False_5_5500.eval()
#
# NOT GP !!
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
#

#
# G_R_H_GP_24_12_6000_2e_06_False_5 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_500.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_500.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_1000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_1000.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_1000.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_1500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_1500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_1500.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_1500.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_2000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_2000.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_2000.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_2500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_2500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_2500.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_2500.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_3000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_3000.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_3000.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_3500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_3500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_3500.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_3500.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_4000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_4000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_4000.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_4000.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_4500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_4500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_4500.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_4500.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_5000 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_5000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_5000.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_5000.eval()
# G_R_H_GP_24_12_6000_2e_06_False_5_5500 = GenZ_1(window_size=24, hidden_size=c, feature_size=7)
# G_R_H_GP_24_12_6000_2e_06_False_5_5500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/R_H/GP_generator_R_H_24_12_6000_2e-06_False_5_5500.pth'))
# G_R_H_GP_24_12_6000_2e_06_False_5_5500.eval()
#




# Z_N
# G_RN_20_1_3000_2e_6_5 = GenZ_N(window_size=K, hidden_size=c, feature_size=F)
# G_RN_20_1_3000_2e_6_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_RN_20_1_3000_2e-06_False_5.pth'))
# G_RN_20_1_3000_2e_6_5.eval()
# G_RN_20_1_3000_2e_6_4 = GenZ_N(window_size=K, hidden_size=c, feature_size=F)
# G_RN_20_1_3000_2e_6_4.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_RN_20_1_3000_2e-06_False_4.pth'))
# G_RN_20_1_3000_2e_6_4.eval()
# G_RN_20_1_3000_2e_6_5_best_1000 = GenZ_N(window_size=K, hidden_size=c, feature_size=F)
# G_RN_20_1_3000_2e_6_5_best_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_RN_20_1_3000_2e-06_False_5_best_1000.pth'))
# G_RN_20_1_3000_2e_6_5_best_1000.eval()
# G_RN_20_1_3000_2e_6_3_best_1000 = GenZ_N(window_size=K, hidden_size=c, feature_size=F)
# G_RN_20_1_3000_2e_6_3_best_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/generator_RN_20_1_3000_2e-06_False_3_best_1000.pth'))
# G_RN_20_1_3000_2e_6_3_best_1000.eval()
# G_RN_20_1_3000_2e_06_False_5_best_1000 = GenZ_N(window_size=K, hidden_size=c, feature_size=F)
# G_RN_20_1_3000_2e_06_False_5_best_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN/generator_RN_20_1_3000_2e-06_False_5_best_1000.pth'))
# G_RN_20_1_3000_2e_06_False_5_best_1000.eval()
# G_RN_20_1_3000_2e_06_False_5_best_2000 = GenZ_N(window_size=K, hidden_size=c, feature_size=F)
# G_RN_20_1_3000_2e_06_False_5_best_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN/generator_RN_20_1_3000_2e-06_False_5_best_2000.pth'))
# G_RN_20_1_3000_2e_06_False_5_best_2000.eval()
# G_RN_20_1_3000_2e_06_False_5_best_3000 = GenZ_N(window_size=K, hidden_size=c, feature_size=F)
# G_RN_20_1_3000_2e_06_False_5_best_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN/generator_RN_20_1_3000_2e-06_False_5_best_3000.pth'))
# G_RN_20_1_3000_2e_06_False_5_best_3000.eval()
# G_RN_H_60_30_1500_2e_06_False_5 = GenZ_N(window_size=60, hidden_size=c, feature_size=12)
# G_RN_H_60_30_1500_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_60_30_1500_2e-06_False_5.pth'))
# G_RN_H_60_30_1500_2e_06_False_5.eval()
# G_RN_H_60_30_1500_2e_06_False_5_best_2000 = GenZ_N(window_size=60, hidden_size=c, feature_size=12)
# G_RN_H_60_30_1500_2e_06_False_5_best_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_60_30_1500_2e-06_False_5_best_2000.pth'))
# G_RN_H_60_30_1500_2e_06_False_5_best_2000.eval()
# G_RN_H_60_30_1500_2e_06_False_5_best_1000 = GenZ_N(window_size=60, hidden_size=c, feature_size=12)
# G_RN_H_60_30_1500_2e_06_False_5_best_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_60_30_1500_2e-06_False_5_best_1000.pth'))
# G_RN_H_60_30_1500_2e_06_False_5_best_1000.eval()
# G_RN_H_24_24_3000_2e_06_False_2 = GenZ_N(window_size=24, hidden_size=c, feature_size=12)
# G_RN_H_24_24_3000_2e_06_False_2.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_24_24_3000_2e-06_False_2.pth'))
# G_RN_H_24_24_3000_2e_06_False_2.eval()
#
# G_RN_H_24_12_6000_2e_06_False_5 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_12_6000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_24_12_6000_2e-06_False_5.pth'))
# G_RN_H_24_12_6000_2e_06_False_5.eval()
# G_RN_H_24_12_6000_2e_06_False_5_500 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_12_6000_2e_06_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_R_H_24_12_6000_2e-06_False_5_500.pth'))
# G_RN_H_24_12_6000_2e_06_False_5_500.eval()
# G_RN_H_24_12_6000_2e_06_False_5_1000 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_12_6000_2e_06_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_R_H_24_12_6000_2e-06_False_5_1000.pth'))
# G_RN_H_24_12_6000_2e_06_False_5_1000.eval()
# G_RN_H_24_12_6000_2e_06_False_5_1500 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_12_6000_2e_06_False_5_1500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_R_H_24_12_6000_2e-06_False_5_1500.pth'))
# G_RN_H_24_12_6000_2e_06_False_5_1500.eval()
# G_RN_H_24_12_6000_2e_06_False_5_2000 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_12_6000_2e_06_False_5_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_R_H_24_12_6000_2e-06_False_5_2000.pth'))
# G_RN_H_24_12_6000_2e_06_False_5_2000.eval()
# G_RN_H_24_12_6000_2e_06_False_5_2500 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_12_6000_2e_06_False_5_2500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_R_H_24_12_6000_2e-06_False_5_2500.pth'))
# G_RN_H_24_12_6000_2e_06_False_5_2500.eval()
# G_RN_H_24_12_6000_2e_06_False_5_3000 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_12_6000_2e_06_False_5_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_R_H_24_12_6000_2e-06_False_5_3000.pth'))
# G_RN_H_24_12_6000_2e_06_False_5_3000.eval()
# G_RN_H_24_12_6000_2e_06_False_5_3500 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_12_6000_2e_06_False_5_3500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_R_H_24_12_6000_2e-06_False_5_3500.pth'))
# G_RN_H_24_12_6000_2e_06_False_5_3500.eval()
# G_RN_H_24_12_6000_2e_06_False_5_4000 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_12_6000_2e_06_False_5_4000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_R_H_24_12_6000_2e-06_False_5_4000.pth'))
# G_RN_H_24_12_6000_2e_06_False_5_4000.eval()
# G_RN_H_24_12_6000_2e_06_False_5_4500 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_12_6000_2e_06_False_5_4500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_R_H_24_12_6000_2e-06_False_5_4500.pth'))
# G_RN_H_24_12_6000_2e_06_False_5_4500.eval()
# G_RN_H_24_12_6000_2e_06_False_5_5000 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_12_6000_2e_06_False_5_5000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_R_H_24_12_6000_2e-06_False_5_5000.pth'))
# G_RN_H_24_12_6000_2e_06_False_5_5000.eval()
# G_RN_H_24_12_6000_2e_06_False_5_5500 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_12_6000_2e_06_False_5_5500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_R_H_24_12_6000_2e-06_False_5_5500.pth'))
# G_RN_H_24_12_6000_2e_06_False_5_5500.eval()
# #
# G_RN_H_24_24_6000_2e_06_False_5 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_24_6000_2e_06_False_5.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_24_24_6000_2e-06_False_5.pth'))
# G_RN_H_24_24_6000_2e_06_False_5.eval()
# G_RN_H_24_24_6000_2e_06_False_5_500 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_24_6000_2e_06_False_5_500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_24_24_6000_2e-06_False_5_500.pth'))
# G_RN_H_24_24_6000_2e_06_False_5_500.eval()
# G_RN_H_24_24_6000_2e_06_False_5_1000 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_24_6000_2e_06_False_5_1000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_24_24_6000_2e-06_False_5_1000.pth'))
# G_RN_H_24_24_6000_2e_06_False_5_1000.eval()
# G_RN_H_24_24_6000_2e_06_False_5_1500 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_24_6000_2e_06_False_5_1500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_24_24_6000_2e-06_False_5_1500.pth'))
# G_RN_H_24_24_6000_2e_06_False_5_1500.eval()
# G_RN_H_24_24_6000_2e_06_False_5_2000 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_24_6000_2e_06_False_5_2000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_24_24_6000_2e-06_False_5_2000.pth'))
# G_RN_H_24_24_6000_2e_06_False_5_2000.eval()
# G_RN_H_24_24_6000_2e_06_False_5_2500 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_24_6000_2e_06_False_5_2500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_24_24_6000_2e-06_False_5_2500.pth'))
# G_RN_H_24_24_6000_2e_06_False_5_2500.eval()
# G_RN_H_24_24_6000_2e_06_False_5_3000 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_24_6000_2e_06_False_5_3000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_24_24_6000_2e-06_False_5_3000.pth'))
# G_RN_H_24_24_6000_2e_06_False_5_3000.eval()
# G_RN_H_24_24_6000_2e_06_False_5_3500 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_24_6000_2e_06_False_5_3500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_24_24_6000_2e-06_False_5_3500.pth'))
# G_RN_H_24_24_6000_2e_06_False_5_3500.eval()
# G_RN_H_24_24_6000_2e_06_False_5_4000 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_24_6000_2e_06_False_5_4000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_24_24_6000_2e-06_False_5_4000.pth'))
# G_RN_H_24_24_6000_2e_06_False_5_4000.eval()
# G_RN_H_24_24_6000_2e_06_False_5_4500 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_24_6000_2e_06_False_5_4500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_24_24_6000_2e-06_False_5_4500.pth'))
# G_RN_H_24_24_6000_2e_06_False_5_4500.eval()
# G_RN_H_24_24_6000_2e_06_False_5_5000 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_24_6000_2e_06_False_5_5000.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_24_24_6000_2e-06_False_5_5000.pth'))
# G_RN_H_24_24_6000_2e_06_False_5_5000.eval()
# G_RN_H_24_24_6000_2e_06_False_5_5500 = GenZ_N(window_size=24, hidden_size=c, feature_size=7)
# G_RN_H_24_24_6000_2e_06_False_5_5500.load_state_dict(torch.load('C:/Users/D255728/Documents/ProyectoGAN/model/RN_H/generator_RN_H_24_24_6000_2e-06_False_5_5500.pth'))
# G_RN_H_24_24_6000_2e_06_False_5_5500.eval()
#



# real_2019_data = torch.zeros(test_data.shape[0] - K + 1, K, F)
# real_2019_data = torch.zeros(K, F)
# for i in range(real_2019.shape[0] - K + 1):
#     real_2019_data[i, :, :] = test_data[i:i + K, :]
# start = test_data.shape[0] - K
# start = 10
# real_2019_data = test_data[start:start + K, :]

# plt.plot(real_2019_data[:, -1].reshape(-1,), color='r', linestyle='-')

# # print(f'{real_2019_data[:, :-1].shape=}')
# y = real_2019_data[:, :-1].reshape(-1, K, F - 1)
# # print(f'{y.shape=}')

# # z = torch.randn(test_data.shape[0] - K + 1, K, 1)  # Z_1
# samples = 100
# y_samples = torch.zeros(samples, y.shape[1], y.shape[2])
# for i in range(samples):
#     # print(f'{y_samples[i, :, :].shape=}')
#     y_samples[i, :, :] = y

# z = torch.randn(samples, K, 1)  # Z_1
# pred = G_CR_1500(z, y_samples)
# pred = pred.reshape(-1, K, 1)
# for i in range(samples):
#     plt.plot(pred[i, :, -1].reshape(-1).detach().numpy(), color='b', linestyle='-')
# # plot_model(20, F, 1500, CR, color, z, y_samples)

# pred_CR_3000 = G_CR_3000(z, y_samples)
# pred_CR_3000 = pred_CR_3000.reshape(-1, K, 1)
# for i in range(samples):
#     # if i == 0:
#     #     print(pred_CR_3000[i, :, 0])
#     plt.plot(pred_CR_3000[i, :, -1].reshape(-1).detach().numpy(), color='g', linestyle='-')

# start = 300
# real_2019_data = test_data[start:start + K, :]

# plt.plot(real_2019_data[:, -1].reshape(-1,), color='darkred', linestyle='-')

# # print(f'{real_2019_data[:, :-1].shape=}')
# y = real_2019_data[:, :-1].reshape(-1, K, F - 1)
# # print(f'{y.shape=}')

# # z = torch.randn(test_data.shape[0] - K + 1, K, 1)  # Z_1
# samples = 100
# y_samples = torch.zeros(samples, y.shape[1], y.shape[2])
# for i in range(samples):
#     # print(f'{y_samples[i, :, :].shape=}')
#     y_samples[i, :, :] = y

# z = torch.randn(samples, K, 1)  # Z_1
# # pred = G_CR_1500(z, y_samples)
# # pred = pred.reshape(-1, K, 1)
# # for i in range(samples):
# #     plt.plot(pred[i, :, -1].reshape(-1).detach().numpy(), color='b', linestyle='-')
# # # plot_model(20, F, 1500, CR, color, z, y_samples)

# pred_CR_3000 = G_CR_3000(z, y_samples)
# pred_CR_3000 = pred_CR_3000.reshape(-1, K, 1)
# for i in range(samples):
#     # if i == 0:
#     #     print(pred_CR_3000[i, :, 0])
#     plt.plot(pred_CR_3000[i, :, -1].reshape(-1).detach().numpy(), color='darkblue', linestyle='-')


# pred_CR_6000 = G_CR_6000(z, y_samples)
# pred_CR_6000 = pred_CR_6000.reshape(-1, K, 1)
# for i in range(samples):
#     plt.plot(pred_CR_6000[i, :, -1].reshape(-1).detach().numpy(), color='purple', linestyle='-')

# pred_CR_20_1_3000 = G_CR_20_1_3000(z, y_samples)
# pred_CR_20_1_3000 = pred_CR_20_1_3000.reshape(-1, K, 1)
# for i in range(samples):
#     plt.plot(pred_CR_20_1_3000[i, :, -1].reshape(-1).detach().numpy(), color='skyblue', linestyle='-')

# pred_CR_20_1_6000 = G_CR_20_1_6000(z, y_samples)
# pred_CR_20_1_6000 = pred_CR_20_1_6000.reshape(-1, K, 1)
# for i in range(samples):
#     plt.plot(pred_CR_20_1_6000[i, :, -1].reshape(-1).detach().numpy(), color='teal', linestyle='-')

# pred_CR_20_1_3000_2e_6 = G_CR_20_1_3000_2e_6(z, y_samples)
# pred_CR_20_1_3000_2e_6 = pred_CR_20_1_3000_2e_6.reshape(-1, K, 1)
# for i in range(samples):
#     plt.plot(pred_CR_20_1_3000_2e_6[i, :, -1].reshape(-1).detach().numpy(), color='blueviolet', linestyle='-')

# pred_CR_20_1_3000_2e_5 = G_CR_20_1_3000_2e_5(z, y_samples)
# pred_CR_20_1_3000_2e_5 = pred_CR_20_1_3000_2e_5.reshape(-1, K, 1)
# for i in range(samples):
#     plt.plot(pred_CR_20_1_3000_2e_5[i, :, -1].reshape(-1).detach().numpy(), color='mediumorchid', linestyle='-')

# pred_CR_20_1_3000_2e_5_2 = G_CR_20_1_3000_2e_5_2(z, y_samples)
# pred_CR_20_1_3000_2e_5_2 = pred_CR_20_1_3000_2e_5_2.reshape(-1, K, 1)
# for i in range(samples):
#     plt.plot(pred_CR_20_1_3000_2e_5_2[i, :, -1].reshape(-1).detach().numpy(), color='magenta', linestyle='-')

# #========= K = 30 ==============#
# K = 30

# real_2019_data_k30 = torch.zeros(K, F)
# real_2019_data_k30 = test_data[start:start + K, :]

# y_k30 = real_2019_data_k30[:, :-1].reshape(-1, K, F - 1)
# y_samples_k30 = torch.zeros(samples, y_k30.shape[1], y_k30.shape[2])
# for i in range(samples):
#     y_samples_k30[i, :, :] = y_k30

# z_k30 = torch.randn(samples, K, 1)  # Z_1
# pred_CR_30_1_6000 = G_CR_30_1_6000(z_k30, y_samples_k30)
# pred_CR_30_1_6000 = pred_CR_30_1_6000.reshape(-1, K, 1)
# for i in range(samples):
#     plt.plot(pred_CR_30_1_6000[i, :, -1].reshape(-1).detach().numpy(), color='darkred', linestyle='-')

# #========= K = 10 ==============#
# K = 10

# real_2019_data_k10 = torch.zeros(K, F)
# real_2019_data_k10 = test_data[start:start + K, :]

# y_k10 = real_2019_data_k10[:, :-1].reshape(-1, K, F - 1)
# y_samples_k10 = torch.zeros(samples, y_k10.shape[1], y_k10.shape[2])
# for i in range(samples):
#     y_samples_k10[i, :, :] = y_k10

# z_k10 = torch.randn(samples, K, 1)  # Z_1
# pred_CR_10_5_6000 = G_CR_10_5_6000(z_k10, y_samples_k10)
# pred_CR_10_5_6000 = pred_CR_10_5_6000.reshape(-1, K, 1)
# for i in range(samples):
#     plt.plot(pred_CR_10_5_6000[i, :, -1].reshape(-1).detach().numpy(), color='brown', linestyle='-')

# pred_CR_10_2_6000 = G_CR_10_2_6000(z_k10, y_samples_k10)
# pred_CR_10_2_6000 = pred_CR_10_2_6000.reshape(-1, K, 1)
# for i in range(samples):
#     plt.plot(pred_CR_10_2_6000[i, :, -1].reshape(-1).detach().numpy(), color='yellow', linestyle='-')

# pred_CR_10_1_6000 = G_CR_10_1_6000(z_k10, y_samples_k10)
# pred_CR_10_1_6000 = pred_CR_10_1_6000.reshape(-1, K, 1)
# for i in range(samples):
#     plt.plot(pred_CR_10_1_6000[i, :, -1].reshape(-1).detach().numpy(), color='black', linestyle='-')

#========= K = 10 ==============#


# pred_CR_____ = G_CR_____(z, y_samples)
# pred_CR_____ = pred_CR_____.reshape(-1, K, 1)
# for i in range(samples):
#     plt.plot(pred_CR_____[i, :, -1].reshape(-1).detach().numpy(), color='orange', linestyle='-')


# #========= K = 20 ==============#
# K = 20

# Z_1
# gen_data = {}
# counter = 0
# while len(gen_data) == 0:
#     num_mh_samples = 10_000
#     z_i = torch.randn(num_mh_samples, K, 1)
#     x_pred = G_CR_____(z_i).reshape(-1, K, F)  # (num_mh_samples, K, F)
#     error = 100_000

#     for i in range(x_pred.shape[0]):
#         if torch.equal(torch.sort(x_pred[i, :, 0]).values, x_pred[i, :, 0]):
#             new_error = torch.sum((x_pred[i, :, :-1] - y)**2)
#             if new_error not in gen_data:
#                 gen_data[new_error] = i

#     print(f'Counter: {counter} Sorted: {len(gen_data)}')
#     counter += 1

# gen_data = collections.OrderedDict(sorted(gen_data.items()))

# count = 0
# for k, v in gen_data.items():
#     if count == 0 or count == 99:
#         print(f'{k=}')
#     if count == 0:
#         print(x_pred[v, :, 0])
#     plt.plot(x_pred[v, :, -1].reshape(-1).detach().numpy(), color='orange', linestyle='-')
#     count += 1
#     if count == 100:
#         break


# pred_Z1 = G_Z1(z)
# pred_Z1 = pred_Z1.reshape(-1, K, 1)
# for i in range(samples):
#     plt.plot(pred_Z1[i, :, -1].reshape(-1).detach().numpy(), color='yellow', linestyle='-')


# gen_data = {}
# counter = 0
# while len(gen_data) == 0:
#     num_mh_samples = 10_000
#     z_i = torch.randn(num_mh_samples, K, 1)
#     x_pred = G_R_20_1_3000_2e_5(z_i).reshape(-1, K, F)  # (num_mh_samples, K, F)
#     error = 100_000

#     for i in range(x_pred.shape[0]):
#         if torch.equal(torch.sort(x_pred[i, :, 0]).values, x_pred[i, :, 0]):
#             new_error = torch.sum((x_pred[i, :, :-1] - y)**2)
#             if new_error not in gen_data:
#                 gen_data[new_error] = i

#     print(f'Counter: {counter} Sorted: {len(gen_data)}')
#     counter += 1

# gen_data = collections.OrderedDict(sorted(gen_data.items()))

# count = 0
# for k, v in gen_data.items():
#     if count == 0 or count == 99:
#         print(f'{k=}')
#     if count == 0:
#         print(x_pred[v, :, 0])
#     plt.plot(x_pred[v, :, -1].reshape(-1).detach().numpy(), color='orange', linestyle='-')
#     count += 1
#     if count == 100:
#         break


# gen_data = {}
# num_mh_samples = 10_000
# z_i = torch.randn(num_mh_samples, K, 1)
# x_pred = G_R_20_1_1500_2e_5(z_i).reshape(-1, K, F)  # (num_mh_samples, K, F)

# for i in range(x_pred.shape[0]):
#     # _, idxs = torch.sort(x_pred[i, :, :], dim=0)
#     # a = x_pred[i, idxs[:, 0], :]
#     # error = torch.sum((a[:, :-1] - y)**2)
#     error = torch.sum((x_pred[i, :, :-1] - y)**2)
#     # gen_data[error] = a
#     gen_data[error] = i

# gen_data = collections.OrderedDict(sorted(gen_data.items()))
# # gen_data = dict(sorted(gen_data.items(), key=operator.itemgetter(1), reverse=False))

# count = 0
# for k, v in gen_data.items():
#     plt.plot(x_pred[v, :, -1].reshape(-1).detach().numpy(), color='orange', linestyle='-')
#     # plt.plot(v[:, -1].reshape(-1).detach().numpy(), color='orange', linestyle='-')
#     count += 1
#     if count == 100:
#         break


# gen_data = {}
# num_mh_samples = 20_000
# z_i = torch.randn(num_mh_samples, K, 1)
# x_pred = G_R_20_1_1500_2e_6(z_i).reshape(-1, K, F)  # (num_mh_samples, K, F)

# for i in range(x_pred.shape[0]):
#     error = torch.sum((x_pred[i, :, :-1] - y)**2)
#     gen_data[error] = i

# gen_data = collections.OrderedDict(sorted(gen_data.items()))
# # gen_data = dict(sorted(gen_data.items(), key=operator.itemgetter(1), reverse=False))

# count = 0
# for k, v in gen_data.items():
#     plt.plot(x_pred[v, :, -1].reshape(-1).detach().numpy(), color='darkolivegreen', linestyle='-')
#     count += 1
#     if count == 100:
#         break


# Z_1

# #========= K = 60 , F = 12 ==============#
# K = 60
# F = 12
# #========= K = 24 , F = 12 ==============#
# K = 24
# F = 12
# num_mh_samples = 1_000
# z_i = torch.randn(num_mh_samples, K, 1)
# x_pred = G_R_H_24_24_1500_2e_06_False_5_best_1000_1_layer(z_i).reshape(-1, K, F)  # (num_mh_samples, K, F)

# for j in range(num_mh_samples):
#     # plt.plot(x_pred[j, :, 0].detach().numpy(), color='r', linestyle='-')
#     plt.plot(x_pred[j, :, -1].reshape(-1).detach().numpy(), color='b', linestyle='-')

# best_error = 100_000
# best_x = None
# best_pred_x = None
# for i in range(data.shape[0]):
#     for j in range(num_mh_samples):
#         # print(f'< 0 :{torch.sum(x_pred[j, :, :] < 0)}')
#         print(f'{(torch.sum(x_pred[j, :, :] < 0) + torch.sum(x_pred[j, :, :] > 1)) * 100 / (K * F)}%')
#         # if torch.all(x_pred[j, :, :] >= 0) and torch.all(x_pred[j, :, :] <= 1):
#         #     if x_pred[j, :, :-1].shape[0] == data[i:i + K, :-1].shape[0]:
#         #         error = torch.sum((x_pred[j, :, :-1] - data[i:i + K, :-1])**2)
#         #         if error < best_error:
#         #             best_error = error
#         #             best_x = data[i:i + K, :]
#         #             best_pred_x = x_pred[j, :, :]
#     break
#     if i % 100 == 0:
#         print(f'{i=} {best_error=}')

# print(f'{best_error=}')
# plt.plot(best_x[:, -1].reshape(-1).detach().numpy(), color='r', linestyle='-')
# plt.plot(best_pred_x[:, -1].reshape(-1).detach().numpy(), color='b', linestyle='-')


# Z_N
# #========= K = 24 , F = 12 ==============#
# K = 24
# F = 12

# num_mh_samples = 1_000
# z_i = torch.randn(num_mh_samples, K, F)
# x_pred = G_RN_H_24_24_3000_2e_06_False_2(z_i).reshape(-1, K, F)  # (num_mh_samples, K, F)

# for j in range(num_mh_samples):
#     # print(f'{x_pred[j, :, :]=}')
#     # break
#     # plt.plot(x_pred[j, :, 0].detach().numpy(), color='r', linestyle='-')
#     plt.plot(x_pred[j, :, -1].reshape(-1).detach().numpy(), color='b', linestyle='-')


# #========= K = 60 , F = 12 ==============#
# K = 60
# F = 12

# num_mh_samples = 1_000
# z_i = torch.randn(num_mh_samples, K, 1)
# x_pred = G_R_H_60_30_1500_2e_06_False_5_best_2000(z_i).reshape(-1, K, F)  # (num_mh_samples, K, F)

# for j in range(num_mh_samples):
#     # plt.plot(x_pred[j, :, 0].detach().numpy(), color='r', linestyle='-')
#     plt.plot(x_pred[j, :, -1].reshape(-1).detach().numpy(), color='b', linestyle='-')



# good_samples = []
# i = 0
# while len(good_samples) < 100:
#     print(f'{i=} {len(good_samples)=}')
#     i += 1
#     z_i = torch.randn(num_mh_samples, K, 1)
#     x_pred = G_R_H_24_7_1500_2e_06_False_5_best_2000(z_i).reshape(-1, K, F)  # (num_mh_samples, K, F)
#     for j in range(num_mh_samples):
#         if (-0.1 <= x_pred[j, :, -1]).all() and (x_pred[j, :, -1] <= 1.1).all():
#             good_samples.append(x_pred[j, :, -1])
#             if len(good_samples) == 100:
#                 break

# for j in range(100):
#     plt.plot(good_samples[j].reshape(-1).detach().numpy(), linestyle='-')
# for j in len(num_mh_samples):
#     # plt.plot(x_pred[j, :, 0].detach().numpy(), color='r', linestyle='-')
#     plt.plot(x_pred[j, :, -1].reshape(-1).detach().numpy(), color='b', linestyle='-')

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


# G = G_R_H_24_12_6000_2e_06_False_5
G = G_R_H_24_24_6000_2e_06_False_5
# D = D_R_H_24_12_6000_2e_06_False_5
D = D_R_H_24_24_6000_2e_06_False_5

# print(f'{X.shape=}')
# X_fake = G_R_H_24_12_6000_2e_06_False_5.double()(X)
z_i = torch.randn(X.shape[0], K, 1)
# X_fake = G_R_H_24_12_6000_2e_06_False_5(z_i).reshape(-1, K, F)
X_fake = G(z_i).reshape(-1, K, F)
# print(f'{X_fake.shape=}')
D.double().calibrate_discriminator(X, X_fake)
# D_R_H_24_12_6000_2e_06_False_5.double().calibrate_discriminator(X, X_fake)
num_mh_samples = 3500
# num_mh_samples = 10_000
init_sample_idx = random.randint(0, train_data.shape[0] - 1)
init_sample = torch.tensor(train_data.values[init_sample_idx:init_sample_idx + window, :])
# x_pred = generate_mh_samples(G_R_H_24_12_6000_2e_06_False_5, D_R_H_24_12_6000_2e_06_False_5, num_mh_samples, init_sample).reshape(-1, K, F)
# x_pred = generate_mh_samples(G_R_H_24_12_6000_2e_06_False_5, D_R_H_24_12_6000_2e_06_False_5, num_mh_samples, init_sample)
x_pred = generate_mh_samples(G, D, num_mh_samples, init_sample)
# print(f'{x_pred.shape=}')
# x_pred = torch.tensor(x_pred)

# z_i = torch.randn(num_mh_samples, K, 1)
# x_pred = G_R_H_24_24_6000_2e_06_False_5(z_i).reshape(-1, K, F)  # (num_mh_samples, K, F)

# step = 24
# real_data = torch.zeros((train_data.shape[0] // window, 1))
real_data = []
for testIdx in range(0, train_data.shape[0], step):
    plt.plot(train_data.values[testIdx:testIdx + window, -1], color='r', linestyle='-')
    real_data.append(train_data.values[testIdx:testIdx + window, -1])
np_real_data = np.array(real_data)

fake_data = []
for j in range(num_mh_samples):
    # plt.plot(torch.sum(x_pred[j, :, :], dim=1).detach().numpy(), color='purple', linestyle='-')
    # plt.plot(x_pred[j, :, 0].detach().numpy(), color='darkgreen', linestyle='-')
    # plt.plot(x_pred[j, :, 1].detach().numpy(), color='orange', linestyle='-')
    # plt.plot(x_pred[j, :, -1].detach().numpy(), color='b', linestyle='-')
    plt.plot(x_pred[j, :, -1], color='b', linestyle='-')
    fake_data.append(x_pred[j, :, -1])
np_fake_data = np.array(fake_data)


print(f'{np_real_data.shape=}')
print(f'{np_fake_data.shape=}')

kde = KernelDensity().fit(np_fake_data)
# kde = KernelDensity().fit(np_real_data)

model_log_density = kde.score_samples(np_fake_data)
threshold = np.percentile(model_log_density, 5)
real_points_log_density = kde.score_samples(np_real_data)
ratio_not_covered = np.mean(real_points_log_density <= threshold)
C = 1 - ratio_not_covered

print(f'{C=}')


# data = read_csv('C:/Users/D255728/Documents/ProyectoGAN/datos/test.csv', index_col=0)
# data[data.columns] = MinMaxScaler().fit_transform(data[data.columns])

# min_error = 100
# i = 0

# for sample in good_samples:
#     for K in range(len(data['Demanda']) - 24):
#         if len(data.loc[K:K + 23, :]) >= 24:
#             if min_error > (torch.sum(sample - torch.tensor(data.loc[K:K + 23, 'Demanda'].values))**2)**.5:
#                 min_error = (torch.sum(sample - torch.tensor(data.loc[K:K + 23, 'Demanda'].values))**2)**.5
#                 best_match = torch.tensor(data.loc[K:K + 23, 'Demanda'].values)
#                 best_sample = sample
#     print(f'Paso: {i} Min_Error:{min_error.item()}')
#     i += 1

# plt.plot(best_sample.reshape(-1).detach().numpy(), color='r', linestyle='-')
# plt.plot(best_match.reshape(-1).detach().numpy(), color='b', linestyle='-')

# testIdx = 0
# plt.plot(data.values[testIdx:testIdx + 24, -1], color='r', linestyle='-')
# testIdx = 24
# plt.plot(data.values[testIdx:testIdx + 24, -1], color='r', linestyle='-')
# testIdx = 48
# plt.plot(data.values[testIdx:testIdx + 24, -1], color='r', linestyle='-')
# testIdx = 72
# plt.plot(data.values[testIdx:testIdx + 24, -1], color='r', linestyle='-')
# testIdx = 96
# plt.plot(data.values[testIdx:testIdx + 24, -1], color='r', linestyle='-')
# testIdx = 120
# plt.plot(data.values[testIdx:testIdx + 24, -1], color='r', linestyle='-')
window = 24
step = 12
# window = 60
# step = 30
# step = 24
# for testIdx in range(0, data.shape[0], step):
#     plt.plot(data.values[testIdx:testIdx + window, -1], color='r', linestyle='-')
#     # plt.plot(data.values[testIdx:testIdx + 24, -1], color='r', linestyle='-')
#     # plt.plot(np.sum(data.values[testIdx:testIdx + 24, :], axis=1), color='skyblue', linestyle='-')



# #========= K = 24 , F = 12 ==============#
# K = 24
# F = 12
# #========= K = 24 , F = 7 ==============#
K = 24
F = 7
# # ========= K = 24 , F = 1 ==============#
# K = 24
# F = 1
# # ========= K = 24 , F = 1 ==============#
# K = 60
# F = 7


# num_mh_samples = 1_000
# z_i = torch.randn(num_mh_samples, K, 1)
# x_pred = G_R_H_24_24_6000_2e_06_False_5(z_i).reshape(-1, K, F)  # (num_mh_samples, K, F)

# for j in range(num_mh_samples):
#     # plt.plot(torch.sum(x_pred[j, :, :], dim=1).detach().numpy(), color='purple', linestyle='-')
#     # plt.plot(x_pred[j, :, 0].detach().numpy(), color='darkgreen', linestyle='-')
#     # plt.plot(x_pred[j, :, 1].detach().numpy(), color='orange', linestyle='-')
#     plt.plot(x_pred[j, :, -1].detach().numpy(), color='b', linestyle='-')


# # num_mh_samples = 10_000
# num_mh_samples = 1000
# # num_mh_samples = 10
# z_i = torch.randn(num_mh_samples, K, 1)
# # x_pred = G_R_H_24_12_6000_2e_06_False_5(z_i).reshape(-1, K, F)
# x_pred = G_R_H_24_24_6000_2e_06_False_5_5000(z_i).reshape(-1, K, F)
# # x_pred = G_R_H_60_30_6000_2e_06_False_5(z_i).reshape(-1, K, F)

# print(f'Window: {window}, Step: {step}, Samples: {num_mh_samples // 10}, Seed: {seed}')

# bestError = 1000
# bestMatch = None
# bestJ = None
# for j in range(num_mh_samples):
#     for testIdx in range(0, data.shape[0], step):
#         if data.values[testIdx:testIdx + window, -1].shape[0] == window:
#             error = np.sqrt(np.sum((x_pred[j, :, -1].detach().numpy() - data.values[testIdx:testIdx + window, -1])**2))
#             if error < bestError:
#                 bestError = error
#                 bestMatch = data.values[testIdx:testIdx + window, :]
#                 bestJ = j

# print(f'{bestError=}')
# print(f'{bestJ=}')

# plt.plot(bestMatch[:, -1], color='r', linestyle='-')
# plt.plot(x_pred[bestJ, :, -1].detach().numpy(), color='b', linestyle='-')
# # plt.plot(bestMatch[:, 0], color='orange', linestyle='-')
# # plt.plot(x_pred[bestJ, :, -2].detach().numpy(), color='darkgreen', linestyle='-')


# start_time = time.time()
# best = {}
# bestError = 1000
# bestMatch = None
# bestJ = None
# bestIdx = None
# for j in range(num_mh_samples):
#     for testIdx in range(0, data.shape[0], step):
#         if data.values[testIdx:testIdx + window, -1].shape[0] == window:
#             # error = np.sqrt(np.sum((x_pred[j, :, -1].detach().numpy() - data.values[testIdx:testIdx + window, -1])**2))
#             # error = np.sum(np.abs(x_pred[j, :, -1].detach().numpy() - data.values[testIdx:testIdx + window, -1]))
#             if sum(data.values[testIdx:testIdx + window, -1] == 0) == 0:
#                 error = (100 / window) * np.sum(np.abs((x_pred[j, :, -1].detach().numpy() - data.values[testIdx:testIdx + window, -1]) / data.values[testIdx:testIdx + window, -1]))
#                 # print(f'{error=}')
#                 if error < bestError:
#                     bestError = error
#                     bestMatch = data.values[testIdx:testIdx + window, :]
#                     bestIdx = testIdx
#                     bestJ = j
#     best[j] = (bestError, bestMatch, bestJ, bestIdx)
#     bestError = 1000
#     bestMatch = None
#     bestIdx = None
#     bestJ = None
#     if j % 100 == 0:
#         print(f'Sample: {j} Time: {(time.time() - start_time)/60:.1f}m')

# print(max(Counter([x[3] for x in best.values()]).values()))

# # print(best)
# # top = 1000
# # # top = 100
# # # top = 2
# top = num_mh_samples // 10

# best = list(dict(sorted(best.items(), key=lambda x: x[1][0])).items())[:top]
# # print(best)

# print(max(Counter([x[1][3] for x in best]).values()))
# heter = len(Counter([x[1][3] for x in best]))

# for j in range(top):
#     plt.plot(best[j][1][1][:, -1], color='r', linestyle='-')
#     plt.plot(x_pred[best[j][1][2], :, -1].detach().numpy(), color='b', linestyle='-')

# # print(f'{best.shape=}')
# # print(f'MSE: {np.mean([x[1][0] for x in best])}')
# print(f'MAPE: {np.mean([x[1][0] for x in best]):.3f}%')
# print(f'HETER: {100 * heter / top}%')

############## RN ################

# window = K = 24
# F = 7
# # step = 12
# # window = 60
# # step = 30
# # step = 12
# for testIdx in range(0, data.shape[0], step):
#     plt.plot(data.values[testIdx:testIdx + window, -1], color='r', linestyle='-')

# num_mh_samples = 1_000
# z_i = torch.randn(num_mh_samples, K, F)
# x_pred = G_RN_H_24_24_6000_2e_06_False_5(z_i).reshape(-1, K, F)  # (num_mh_samples, K, F)

# for j in range(num_mh_samples):
#     # plt.plot(torch.sum(x_pred[j, :, :], dim=1).detach().numpy(), color='purple', linestyle='-')
#     # plt.plot(x_pred[j, :, 0].detach().numpy(), color='darkgreen', linestyle='-')
#     # plt.plot(x_pred[j, :, 1].detach().numpy(), color='orange', linestyle='-')
#     plt.plot(x_pred[j, :, -1].detach().numpy(), color='b', linestyle='-')


# num_mh_samples = 1000
# z_i = torch.randn(num_mh_samples, K, F)
# x_pred = G_RN_H_24_12_6000_2e_06_False_5(z_i).reshape(-1, K, F)
# # for j in range(num_mh_samples):

# bestError = 100
# bestMatch = None
# bestJ = None
# for j in range(num_mh_samples):
#     for testIdx in range(0, data.shape[0], step):
#         error = np.sqrt(np.sum((x_pred[j, :, -1].detach().numpy() - data.values[testIdx:testIdx + window, -1])**2))
#         if error < bestError:
#             bestError = error
#             bestMatch = data.values[testIdx:testIdx + window, :]
#             bestJ = j

# print(f'{bestError=}')
# print(f'{bestJ=}')

# plt.plot(bestMatch[:, -1], color='r', linestyle='-')
# plt.plot(x_pred[bestJ, :, -1].detach().numpy(), color='b', linestyle='-')
# # plt.plot(bestMatch[:, 0], color='orange', linestyle='-')
# # plt.plot(x_pred[bestJ, :, -2].detach().numpy(), color='darkgreen', linestyle='-')


############## RN ################





############## CR ################
# split_idx = 78888
# train_data = data[0:split_idx]
# test_data = data[split_idx:]

# # num_mh_samples = 1000
# # z_i = torch.randn(num_mh_samples, K, 1)
# # x_pred = G_R_H_GP_24_24_6000_2e_06_False_5(z_i).reshape(-1, K, F)

# window = 60
# K = 60

# testIdx = 0
# y = test_data.values[testIdx:testIdx + window, :-1]

# # z = torch.randn(test_data.shape[0] - K + 1, K, 1)  # Z_1
# samples = 1000
# y_samples = torch.zeros(samples, y.shape[0], y.shape[1])
# # for i in range(samples):
# #     # print(f'{y_samples[i, :, :].shape=}')
# #     y_samples[i, :, :] = y[:,:]
# y_samples = torch.tensor(y).repeat(samples, 1, 1)

# # print(f'{y_samples.dtype=}')

# z = torch.randn(samples, K, 1)  # Z_1
# # print(f'{z.double().dtype=}')
# z = z.double()
# pred = G_CR_60_30_6000_2e_6_False_5_4000.double()(z, y_samples)
# pred = pred.reshape(-1, K, 1)
# for i in range(samples):
#     plt.plot(pred[i, :, -1].reshape(-1).detach().numpy(), color='b', linestyle='-')
# # plot_model(20, F, 1500, CR, color, z, y_samples)


# plt.plot(test_data.values[testIdx:testIdx + window, -1], color='r')
############## CR ################


# plt.ylim([-0.1, 1.1])



# #========= K = 60 , F = 12 ==============#
# K = 60
# F = 12

# num_mh_samples = 1_000
# z_i = torch.randn(num_mh_samples, K, F)
# x_pred = G_RN_H_60_30_1500_2e_06_False_5_best_2000(z_i).reshape(-1, K, F)  # (num_mh_samples, K, F)

# for j in range(num_mh_samples):
#     # print(f'{x_pred[j, :, :]=}')
#     # break
#     # plt.plot(x_pred[j, :, 0].detach().numpy(), color='r', linestyle='-')
#     plt.plot(x_pred[j, :, -1].reshape(-1).detach().numpy(), color='b', linestyle='-')


# #========= K = 60 , F = 12 ==============#
# K = 60
# F = 12


# idxTest = 690
# # z_cond = torch.zeros(num_mh_samples, K, F - 1)
# # for i in range(num_mh_samples):
# #     z_cond[i, :, :] = torch.from_numpy(data.values[idxTest:idxTest + K, :-1]).float()
# z_cond = torch.tensor(data.values[idxTest:idxTest + K, :-1])
# z_cond = z_cond.repeat(num_mh_samples, 1, 1)

# num_mh_samples = 1_000
# z_rand = torch.randn(num_mh_samples, K, 1)
# # z_rand = z_rand.double()

# # z_i = torch.cat((z_cond, z_rand), dim=2)
# # print(f'{z_cond.dtype=}')
# # print(f'{z_rand.dtype=}')
# # print(f'{z_rand.double().dtype=}')

# x_pred = G_CR_60_30_1500_2e_6_False_5(z_rand, z_cond.float()).reshape(-1, K, 1)  # (num_mh_samples, K, F)

# plt.plot(torch.tensor(data.values[idxTest:idxTest + K, -1]), color='r', linestyle='-')

# for j in range(num_mh_samples):
#     # print(f'{x_pred[j, :, :]=}')
#     # break
#     # plt.plot(x_pred[j, :, 0].detach().numpy(), color='r', linestyle='-')
#     plt.plot(x_pred[j, :, -1].reshape(-1).detach().numpy(), color='b', linestyle='-')


# #========= K = 24 , F = 24 ==============#
# K = 24
# F = 24


# idxTest = 420
# num_mh_samples = 1_000
# # z_cond = torch.zeros(num_mh_samples, K, F - 1)
# # for i in range(num_mh_samples):
# #     z_cond[i, :, :] = torch.from_numpy(data.values[idxTest:idxTest + K, :-1]).float()
# z_cond = torch.tensor(data.values[idxTest:idxTest + K, :-1])
# z_cond = z_cond.repeat(num_mh_samples, 1, 1)

# z_rand = torch.randn(num_mh_samples, K, 1)
# # z_rand = z_rand.double()

# # z_i = torch.cat((z_cond, z_rand), dim=2)
# # print(f'{z_cond.dtype=}')
# # print(f'{z_rand.dtype=}')
# # print(f'{z_rand.double().dtype=}')

# x_pred = G_CR_24_24_1500_2e_6_False_5(z_rand, z_cond.float()).reshape(-1, K, 1)  # (num_mh_samples, K, F)

# plt.plot(torch.tensor(data.values[idxTest:idxTest + K, -1]), color='r', linestyle='-')

# for j in range(num_mh_samples):
#     # print(f'{x_pred[j, :, :]=}')
#     # break
#     # plt.plot(x_pred[j, :, 0].detach().numpy(), color='r', linestyle='-')
#     plt.plot(x_pred[j, :, -1].reshape(-1).detach().numpy(), color='b', linestyle='-')


# plt.ylim([-0.1, 5])
# plt.ylim([-0.1, 1.1])
plt.show()

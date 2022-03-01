# ProyectoGAN

Proyecto sobre demanda energética Uruguaya utilizando redes adversarias generativas (GANs)

<br/>
Para entrenar el modelo horario correr el comando:

python R_GAN_H.py window_size step_size epochs n_critic

por ejemplo:
`python R_GAN_H.py 24 24 6000 5`

<br/>
Para evaluar el modelo correr el comando (genera num_samples datos sintéticos y calcula la métrica de Coverage):

python evalGAN.py num_samples

por ejemplo:
`python evalGAN.py 3500`

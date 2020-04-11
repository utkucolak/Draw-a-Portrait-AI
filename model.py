from tensorflow.keras.layers import Input,SimpleRNN,GRU,LSTM,Dense,Flatten,GlobalMaxPooling1D,Embedding, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from cv2 import imread
import cv2
img_dir = "C:\\Users\\Casper\\Desktop\\draw_ata\\new_data\\"
data_train = os.listdir(img_dir)[:10000]
data_test = os.listdir(img_dir)[50000:]

for i in range(len(data_train)):
	data_train[i] = img_to_array(cv2.resize(imread(img_dir + data_train[i],cv2.IMREAD_GRAYSCALE),(150,150)))
data_train = np.stack(data_train)
data_train = data_train / 255.0
N, H, W, _ = data_train.shape
D = H*W
data_train = data_train.reshape(-1, D)
latent_dim = 100
def build_generator(latent_dim):
	i = Input(shape=(latent_dim,))
	x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
	x = BatchNormalization(momentum=0.8)(x)
	x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = Dense(D, activation="tanh")(x)
	model = Model(i, x)

	return model

def build_discriminator(img_size):
	i = Input(shape=(img_size,))
	x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
	x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
	x = Dense(1, activation='sigmoid')(x)
	model = Model(i, x)
	return model
discriminator = build_discriminator(D)
discriminator.compile(loss="binary_crossentropy", optimizer=Adam(0.002,0.5),metrics=["accuracy"])
generator = build_generator(latent_dim)
z = Input(shape=(latent_dim,))
img = generator(z)
discriminator.trainable = False
fake_pred = discriminator(img)
combined_model = Model(z,fake_pred)
combined_model.compile(loss="binary_crossentropy", optimizer=Adam(0.002,0.5))
batch_size = 32
epochs=30000
sample_period = 500
ones = np.ones(batch_size)
zeros = np.zeros(batch_size)
d_losses = []
g_losses = []
if not os.path.exists('gan_images'):
	os.makedirs('gan_images')
def sample_images(epoch):
	rows,cols = 5, 5 
	noise = np.random.randn(rows*cols, latent_dim)
	imgs = generator.predict(noise)

	imgs = 0.5 * imgs + 0.5
	fig, axs = plt.subplots(rows,cols)
	idx = 0
	for i in range(rows):
		for j in range(cols):
			axs[i,j].imshow(imgs[idx].reshape(H, W), cmap="gray")
			axs[i,j].axis('off')
			idx += 1
	fig.savefig("gan_images/%d.png" % epoch)
	plt.close()
for epoch in range(epochs):
	idx = np.random.randint(0, data_train.shape[0], batch_size)
	real_imgs = data_train[idx]
	noise = np.random.randn(batch_size, latent_dim)
	fake_imgs = generator.predict(noise)

	d_loss_real, d_acc_real = discriminator.train_on_batch(real_imgs, ones)
	d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)
	d_loss = 0.5 * (d_loss_fake + d_loss_real)
	d_acc = 0.5 * (d_acc_real + d_acc_fake)

	noise = np.random.randn(batch_size, latent_dim)
	g_loss = combined_model.train_on_batch(noise,ones)
	noise = np.random.randn(batch_size, latent_dim)
	g_loss = combined_model.train_on_batch(noise,ones)
	d_losses.append(d_loss)
	g_losses.append(g_loss)
	if epochs % 100 == 0:
		print(f"epoch: {epoch+1} / {epochs}, d_loss: {d_loss:.2f}, d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}")
	if epoch % sample_period == 0:
		sample_images(epoch)

plt.plot(g_losses)
plt.plot(d_losses)
plt.show()


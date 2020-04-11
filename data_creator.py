import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.image as mpimg
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
img_dir = "C:\\Users\\Casper\\Desktop\\draw_ata\\atam\\"
image_shape = (150,150,3)
if not os.path.exists('new_data'):
	os.makedirs('new_data')
for i in range(2000):
	random_img_path = img_dir + random.choice(os.listdir(img_dir))
	img = imread(random_img_path, 0)
	
	
	mpimg.imsave("new_data/%d.png" % i, img)

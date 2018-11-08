import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from skimage import io
from skimage.color import rgb2gray

from data_generator import data_generator
dg = data_generator(backgrounds_path='backgrounds/', 
                                   fonts_path='valid_fonts/',
                                   valid_charset_path='valid_charset.txt',
                   background_type = ['real','const','const'])
number_of_iteration = 60000
for num in range(int(number_of_iteration)):
    dg.get_text_image_with_bg()

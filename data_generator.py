#!/usr/bin/env python
# coding: utf-8



from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageOps, ImageStat, ImageEnhance
import numpy as np
import random
import os
import re
from imgaug import augmenters as iaa
import multiprocessing



import logging
from logging import handlers

logger = logging.getLogger(__name__)
handler_stream = logging.StreamHandler() #output to console 
formatter = logging.Formatter("%(asctime)s:::%(module)s:::%(message)s")
handler_stream.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler_stream)


class data_generator(object):
    
    def __init__(self, images_height = 32, image_width = 256, max_string_lenght = 35, 
                 background_type = ['real','const'], dataset_type = 'random', backgrounds_path = './backgrounds/',
                 fonts_path = './valid_fonts/', valid_charset_path = './valid_charset.txt', 
                 text_examples = './text_examples.txt', font_size_bound = (30,18)):
        '''
        background_type - list consist used types of bg, e.g. ['real', 'random', 'const']. 
                    If use 'real', then backgrounds_path should be set to used backgrounds.
        dataset - type of dataset 'random' or 'samples'. If 'samples' then text_examples
                    should be defined. If 'random' then text string will be generated from valid 
                    charset randomly.
        valid_char_set - path to text file with valid chars.
        text_examples - file of text string examples
        backgrounds_path - path with background images
        font_size_bound - bounds of font size range, from which value will be chosen
        '''
        assert set(['real', 'random', 'const']).intersection(set(background_type)) == set(background_type), 'background_type argument should be list consisting only "real", "random" or "const"'
        assert dataset_type in ['random', 'samples'] , 'dataset_type argument should be one of "random" or "samples"'
        self.images_height = images_height
        self.image_width = image_width
        self.max_string_lenght = max_string_lenght
        self.background_type = background_type
        self.dataset_type = dataset_type
        self.backgrounds_files = None
        if 'real' in background_type:
            self.backgrounds_files = (backgrounds_path, self.set_backgrounds(backgrounds_path))
            self.backgrounds = self.cache_backgrounds()
        self.text_samples = None
        if dataset_type == 'samples':
            self.text_samples = self.set_texts_samples(text_examples)
        self.fonts_files = (fonts_path, self.set_fonts(fonts_path))
        logger.info('{} fonts to produce data'.format(len(self.fonts_files[1])))
        self.valid_charset = self.set_valid_charset(valid_charset_path)
        self.char_to_indx = dict(zip(self.valid_charset,
            range(len(self.valid_charset))))
        self.font_size_bound = font_size_bound
        #init augumentation pipline
        self.aug_seq_noise = iaa.Sequential([
            iaa.GaussianBlur(sigma=(0,0.05)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2*255)),
            iaa.AverageBlur(k=((1, 5), (1, 5)))
        ])
        self.aug_seq_distortion = iaa.Sequential([
            iaa.Affine(rotate=(-3,3), 
                order=[1], 
                 scale={"x": (0.8, 1.1), "y": (0.8, 1.1)}, 
                 shear=(-10, 10),
                mode='constant', cval=(0)),
            iaa.PerspectiveTransform(scale=(0, 0.02)),
        ])

    def get_batch(self, batch_size = 64):
        '''
        produce batch of text string images
        '''
        image_batch = []
        string_batch = []
        for _ in range(batch_size):
            image_batch.append(np.array(self.get_text_image()))
            string_batch.append(self.text_string)
        return image_batch, string_batch
    
    def get_text_image_with_bg(self):
        '''
        merge backgroundwith text string image
        '''
        background = self.sample_cached_background()
#        background = self.sample_background()
        text_image = self.get_text_image()
        augumented_text_image = self.aug_seq_distortion.augment_image(np.array(text_image))
        text_image_invert = Image.fromarray(augumented_text_image) #distort only text not background
        background.paste(text_image_invert, box=None, mask=text_image_invert)
        image_with_background = background
        noised_image = self.aug_seq_noise.augment_image(np.array(image_with_background))
        if random.sample([1,2,3],1)[0] == 3: # 33% probability
            noised_image = np.array(ImageOps.invert(Image.fromarray(noised_image))) #distort only text not background
        return noised_image

    def get_image_and_label(self, *args, **kwargs):
        '''
        return pair of image and label
        '''
        image = self.get_text_image_with_bg()
        image = np.expand_dims(np.array(image),-1) # image should have channel
        label = self.string_to_label(self.text_string)
        label = np.array(label) + 1 # as labels will be sparce tensor, should not be zero value
        label = np.pad(label,(0, self.max_string_lenght - len(label)), mode='constant') #padd to max string length
        return image, label
    
    def string_to_label(self, string):
        '''
        transform string of chars to list of indeces
        '''
        label = [self.char_to_indx[s] for s in string]
        return label

    def get_text_image(self):
        '''
        produce one text string image
        '''
        self.text_string = self.sample_string()
        font_size = random.sample(range(np.min(self.font_size_bound), np.max(self.font_size_bound)), 1)[0]
        font = self.sample_font(size=font_size)
        text_size = font.getsize(self.text_string.replace('\n','\\n')) # (width, height)
        while (self.images_height-text_size[1])/2 < 1: #if text heights larger then image heights minus 2 
            font_size -= 1
            font = self.sample_font(size=font_size)
            text_size = font.getsize(self.text_string.replace('\n','\\n')) # (width, height)
            #logger.warn('image heights less then text heights, supress text font size')
        vertical_text_indent = int((self.images_height-text_size[1])/2)*4
        horiaontal_text_indent = int((self.images_height-text_size[1])/2)
        cond = text_size[0] + vertical_text_indent > self.image_width #acquired text is too wide, shrinked must it be. (c) yoda
        temp_image_width = text_size[0] + 2*vertical_text_indent if cond else self.image_width
        text = Image.new("RGBA", size=(temp_image_width, self.images_height), color=(0,0,0,0))
        draw = ImageDraw.Draw(text)
        text_brightness = int(np.random.normal(loc=50,scale=20))
        draw.text((vertical_text_indent, horiaontal_text_indent), self.text_string.replace('\n','\\n'), font=font, fill=(text_brightness, text_brightness, text_brightness, 255))
        if cond:
            text = text.resize((self.image_width, self.images_height), resample = Image.BILINEAR)
        return text
    
    def sample_font(self, size):
        '''
        provide sample of random font
        '''
        font_file = random.sample(self.fonts_files[1], 1)[0]
        fnt = ImageFont.truetype(self.fonts_files[0] + font_file, size)
        return fnt
    
    def sample_string(self,):
        '''
        provide sample of string (random or from dataset)
        '''
        if self.dataset_type == 'random':
            valid_charset = self.valid_charset + list('    ') #add three blank spaces to encrease probobility
            random_string_length = random.randint(2, self.max_string_lenght)
            text_string_indx = [random.randint(0, len(valid_charset)-1) for _ in range(random_string_length)]
            text_string = ''.join([valid_charset[i] for i in text_string_indx])
        elif self.dataset_type == 'sample':
            assert self.text_samples != None, 'self.text_sample is None, cant get examples of text'
            text_string = random.sample(self.text_samples, 1)[0]
        if text_string.strip() == '': #if generated empty (blank space) string
            text_string = ''.join(random.sample(list('1234567890,.'), 12))
            logging.warn('Encounted empty string, substituting with {}'.format(text_string))
        if len(set(text_string.strip()))==1 and list(set(text_string.strip()))[0] in list('._,~`"\|'): #case then string consist only from small chars of same types
            text_string = ''.join(random.sample(list('1234567890,.'), 12))
            logging.warn('Encounted empty string, substituting with {}'.format(text_string))
        return text_string
    
    def augument_image(self, image):
        '''
        apply image augumentation
        '''
        return self.aug_seq.augment_image(image)
    
    def sample_background(self,):
        '''
        provide sample of background image
        '''
        bg_type = random.sample(self.background_type,1)[0]
        brightness_level = int(np.random.normal(loc=190,scale=20))
        if bg_type == 'const':
            background_image = Image.new('L', (self.image_width, self.images_height), (brightness_level))
        elif bg_type == 'real':
            background_file = random.sample(self.backgrounds_files[1], 1)[0]
            background_file_path = self.backgrounds_files[0] + background_file
            background_image = Image.open(background_file_path)
            background_image = background_image.convert('L')
            width, height = background_image.size
            left, upper = width - self.image_width, height - self.images_height
            rndm_left, rndm_upper = random.randint(0, left), random.randint(0, upper)
            background_image = background_image.crop(box=(rndm_left, rndm_upper, 
                                                   rndm_left+self.image_width, rndm_upper+self.images_height))
            stat = ImageStat.Stat(background_image)
            background_image = ImageEnhance.Brightness(background_image).enhance(brightness_level/stat.mean[0])
            
        return background_image

    def sample_cached_background(self,):
        '''
        provide sample of background image from cached backgrounds
        '''
        bg_type = random.sample(self.background_type,1)[0]
        if bg_type == 'const':
            brightness_level = int(np.random.normal(loc=190,scale=20))
            background_image = Image.new('L', (self.image_width, self.images_height), (brightness_level))
        elif bg_type == 'real':
            background_image = random.sample(self.backgrounds, 1)[0]
            width, height = background_image.size
            left, upper = width - self.image_width, height - self.images_height
            rndm_left, rndm_upper = random.randint(0, left), random.randint(0, upper)
            background_image = background_image.crop(box=(rndm_left, rndm_upper,
                                                   rndm_left+self.image_width, rndm_upper+self.images_height))
        return background_image
    
    def set_fonts(self, path):
        '''
        set fonts path
        '''
        assert os.path.isdir(path), 'There is no folder {}'.format(path)
        fonts_path = [f for f in next(os.walk(path))[2] if os.path.splitext(f)[1] == '.ttf']
        assert len(fonts_path) !=0, 'Folder {} is empty'.format(path)
        return fonts_path
    
    def set_backgrounds(self, path):
        '''
        set background images path
        '''
        assert os.path.isdir(path), 'There is no folder {}'.format(path)
        backgrounds_files = [f for f in next(os.walk(path))[2] if os.path.splitext(f)[1] == '.jpg']
        valid_backgrounds_files = []
        for f in backgrounds_files:
            im = Image.open(path+f)
            width, height = im.size
            if width>=self.image_width and height>=self.images_height:
                valid_backgrounds_files.append(f)
        assert len(valid_backgrounds_files) !=0, 'No valid backgounds in {}'.format(path)
        logger.info('{} background files to produce data'.format(len(valid_backgrounds_files)))
        return valid_backgrounds_files
    
    def cache_backgrounds(self):
        '''
        cache background images to memory, for acceleration
        '''
        logger.info('Caching background images...')
        path, files = self.backgrounds_files
        images = []
        for f in files:
            images.append(Image.open(path+f).convert('L'))
        return images
    
    def set_texts_samples(self, path):
        '''
        set text file which lines will be used as text strings to draw
        '''
        assert os.path.isfile(path), 'There is no file {}'.format(path)
        with open(path,'r',encoding='utf-8') as f:
            text_samples = r.read().split('\n')
        text_samples = list(set(text_samples))
        assert len(text_samples) != 0, 'text samples file is empty'
        logger.info('{} text strings to produce data'.format(len(text_samples)))
        return text_samples
    
    def set_valid_charset(self, path):
        '''
        set usable chars
        '''
        assert os.path.isfile(path), 'There is no file {}'.format(path)
        with open(path,'r',encoding='utf-8') as f:
            charset_string = f.read()
        valid_charset = list(charset_string)
        assert len(valid_charset) != 0, 'valid charset is empty'
        return valid_charset

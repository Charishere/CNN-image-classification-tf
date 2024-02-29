"""
import data
"""
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib as plt
import keras

def input_files(dir, ratio):
    train_img = []
    train_lbl = []
    test_img = []
    test_lbl = []

    norm_train = []
    norm_test = []

    n = 10
    frame  = n*ratio

    label_dict = {i: j for i, j in enumerate(os.listdir(dir))}
    label_dict.pop(0)
    #label_dict.pop(1)

    photo_paths = []


    for label, path in enumerate(os.listdir(dir)):
        full_path = os.path.join(dir, path)
        if not os.path.isdir(full_path):
            continue
        images = os.listdir(full_path)
        images = images[:n]
        for i, img in enumerate(images):
            image = os.path.join(dir, path, img) 
            if not img.lower().endswith('.ds_store'):           
                img = cv2.imread(os.path.join(dir, path, img))
                img = cv2.resize(img, (1000, 500), interpolation=cv2.INTER_AREA)
                if i < frame:
                    train_lbl.append(label)
                    train_img.append(img)
                    norm_train.append(img.astype('float64'))
                else:
                    test_lbl.append(label)
                    test_img.append(img)
                    photo_paths.append(image)
                    norm_test.append(img.astype('float64'))

    train_img = np.array(train_img)
    train_lbl = np.array(train_lbl)
    test_img = np.array(test_img)
    test_lbl = np.array(test_lbl)
    norm_train = np.array(norm_train)
    norm_test = np.array(norm_test)
    
    print("train data:", 'images:', train_img.shape, " labels:", train_lbl.shape)
    print("test  data:", 'images:', test_img.shape, " labels:", test_lbl.shape)

    train_lbl = tf.keras.utils.to_categorical(train_lbl)
    test_lbl = tf.keras.utils.to_categorical(test_lbl)


    return norm_train, train_lbl, norm_test, test_lbl, label_dict, photo_paths




#filename = '/Users/wuyufei/Desktop/Charlotte/intern/cnn/tf2.0/data1/'
#ratio = 0.3
#norm_train, train_lbl, norm_test, test_lbl, dict, photo_paths_dict = input_files(filename, ratio)
#print(dict)





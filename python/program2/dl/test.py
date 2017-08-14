# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:14:26 2017

@author: lulu
"""

from skimage import io,transform
import tensorflow as tf
import numpy as np
import glob

path='C:/Users/lulu/Desktop/self/python/program2/dl/test/'

pic_dict = {0:'butterfly',1:'camera',2:'scissors',3:'sunflower',4:'others'}

w=100
h=100
c=3


def read_img(path):
    imgs=[]
    for im in glob.glob(path+'*.jpg'):
        print('reading the images:%s'%(im))
        img=io.imread(im)
        img=transform.resize(img,(w,h,3))
        imgs.append(img)
    return np.asarray(imgs,np.float32)


with tf.Session() as sess:
    data=read_img(path)

    saver = tf.train.import_meta_graph('C:/Users/lulu/Desktop/self/python/program2/dl/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('C:/Users/lulu/Desktop/self/python/program2/dl/'))

    
    
    x=tf.placeholder(tf.float32,shape=[31,w,h,c])####
    #x = graph.get_tensor_by_name("x:0")
    
    graph = tf.get_default_graph()
    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict = {x:data})######

    #打印出预测矩阵
    print(classification_result)
    #打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result,1).eval())
    #根据索引通过字典对应花的分类
    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        print("第",i+1,"张图预测:"+pic_dict[output[i]])
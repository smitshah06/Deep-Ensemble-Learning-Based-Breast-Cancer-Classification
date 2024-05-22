# from numba import jit, cuda
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import glob
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# @jit(target_backend='cuda')
def checkLRFlip(image):

    # Get number of rows and columns in the image.
    nrows, ncols = image.shape[:2]
    x_center = ncols // 2
    y_center = nrows // 2

    # Sum down each column.
    col_sum = image.sum(axis=0)
    # Sum across each row.
    row_sum = image.sum(axis=1)

    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:])

    if left_sum > right_sum:
        return False

    return True


def makeLRFlip(img):

    flipped_img = cv2.flip(img,1)
    return flipped_img

def init_grabcut_mask(h, w, s):
    if s == 1:
        mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD
        mask[:14*h//15, :w//10] = cv2.GC_PR_FGD
        mask[:18*h//20, w//10:2*w//10] = cv2.GC_PR_FGD
        mask[2*h//16:14*h//16, w//6:13*w//48] = cv2.GC_PR_FGD
        mask[3*h//20:17*h//20, 13*w//48:15*w//48] = cv2.GC_PR_FGD
        mask[7*h//40:25*h//30, 15*w//48:17*w//48] = cv2.GC_PR_FGD
        mask[9*h//40:24*h//30, 17*w//48:19*w//48] = cv2.GC_PR_FGD
        mask[11*h//40:23*h//30, 19*w//48:21*w//48] = cv2.GC_PR_FGD
        mask[13*h//40:21*h//30, 21*w//48:22*w//48] = cv2.GC_PR_FGD
        mask[15*h//40:20*h//30, 22*w//48:23*w//48] = cv2.GC_PR_FGD

        mask[h//24:16*h//18, :7*w//80] = cv2.GC_FGD
        mask[h//14:12*h//14, 7*w//80:3*w//20] = cv2.GC_FGD
        mask[2*h//12:10*h//12, 3*w//20:5*w//20] = cv2.GC_FGD
        mask[4*h//20:16*h//20, 5*w//20:13*w//40] = cv2.GC_FGD
        mask[5*h//20:23*h//30, 13*w//40:15*w//40] = cv2.GC_FGD
        mask[9*h//30:22*h//30, 15*w//40:16*w//40] = cv2.GC_FGD
        mask[10*h//30:21*h//30, 16*w//40:17*w//40] = cv2.GC_FGD
        mask[11*h//30:20*h//30, 17*w//40:18*w//40] = cv2.GC_FGD
        mask[12*h//30:19*h//30, 18*w//40:37*w//80] = cv2.GC_FGD
    elif s == 2:
        mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD
        mask[h//15:14*h//15, :w//10] = cv2.GC_PR_FGD
        mask[2*h//20:18*h//20, w//10:2*w//10] = cv2.GC_PR_FGD
        mask[2*h//16:14*h//16, w//6:13*w//48] = cv2.GC_PR_FGD
        mask[3*h//20:17*h//20, 13*w//48:15*w//48] = cv2.GC_PR_FGD
        mask[7*h//40:25*h//30, 15*w//48:17*w//48] = cv2.GC_PR_FGD
        mask[9*h//40:24*h//30, 17*w//48:19*w//48] = cv2.GC_PR_FGD
        mask[11*h//40:23*h//30, 19*w//48:21*w//48] = cv2.GC_PR_FGD
        mask[13*h//40:21*h//30, 21*w//48:22*w//48] = cv2.GC_PR_FGD
        mask[15*h//40:20*h//30, 22*w//48:23*w//48] = cv2.GC_PR_FGD

        mask[2*h//18:16*h//18, :7*w//80] = cv2.GC_FGD
        mask[2*h//14:12*h//14, 7*w//80:3*w//20] = cv2.GC_FGD
        mask[2*h//12:10*h//12, 3*w//20:5*w//20] = cv2.GC_FGD
        mask[4*h//20:16*h//20, 5*w//20:13*w//40] = cv2.GC_FGD
        mask[5*h//20:23*h//30, 13*w//40:15*w//40] = cv2.GC_FGD
        mask[9*h//30:22*h//30, 15*w//40:16*w//40] = cv2.GC_FGD
        mask[10*h//30:21*h//30, 16*w//40:17*w//40] = cv2.GC_FGD
        mask[11*h//30:20*h//30, 17*w//40:18*w//40] = cv2.GC_FGD
        mask[12*h//30:19*h//30, 18*w//40:37*w//80] = cv2.GC_FGD
    return mask

# @jit(target_backend='cuda')
def add_contours(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # cv2.drawContours(image, contours, -1, (255, 0, 0), 3)
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0) ,2)

    return x,y,w,h

# @jit(target_backend='cuda')
def remove_background(image,image1,n,p,img_name, loop):
    h, w = image.shape[:2]
    mask = init_grabcut_mask(h, w, loop)
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bgm, fgm, 4, cv2.GC_INIT_WITH_MASK)
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = cv2.bitwise_and(image, image1, mask = mask_binary)
    x,y,w,h = add_contours(result, mask_binary) # optional, adds visualizations
    result1 = image[y:y+h, x:x+w]
    cv2.imwrite(img_name,result1)

from threading import Thread

dir = '/home/bce19229/19bce229/Dataset/CDS/Class-Divided-SET/'
st_dir = '/home/bce19229/19bce229/Dataset/CDS/Cropped-CDS/'
n = 0
d = 1
dir_list = os.listdir(dir)
dir_list.sort()

with tqdm(total=len(dir_list[1:])) as pbar:
    for class_addr in dir_list[1:]:
        dir_image_list = os.listdir(dir+class_addr)
        with tqdm(total=len(dir_image_list)) as ipbar:
            for image_addr in dir_image_list:
                for i in range(1,3):
                    img_name = st_dir + class_addr + '/' + image_addr
                    addr = dir + class_addr+'/'+image_addr
                    img = cv2.imread(addr)
                    flip_check = cv2.imread(addr, 0)
                    # src = cv2.cuda_GpuMat()
                    # src.upload(img)
                    if i == 1:
                        if checkLRFlip(flip_check):
                            img = makeLRFlip(img)
                    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    lab= cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl,a,b))
                    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                    t = Thread(target=remove_background(im,final,n,d,img_name,i), args=())
                    t.daemon = True
                    t.start()
                    n+=1
                ipbar.update(1)
        pbar.update(1)
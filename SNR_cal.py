import os
import numpy as np
import math
import cv2
def cal_snr(noise_img,cleam_img):
    noise_signal = noise_img-cleam_img
    clean_signal = clean_img
    noise_signal_2 = noise_signal*noise_signal
    clean_signal_2 = clean_signal*clean_signal
    sum_1 = np.sum(noise_signal_2)
    sum_2 = np.sum(clean_signal_2)
    snrr  = 20*math.log10(math.sqrt(sum_1)/math.sqrt(sum_2))
    return snrr

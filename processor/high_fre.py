import numpy as np
import cv2
import torch
import os
import torchvision.transforms as transforms
import torch.nn as nn
import random
def gaussian_high_pass_filter(fshift, D):
    h, w = fshift.shape
    y, x = np.ogrid[:h, :w]
    center = (h / 2, w / 2)
    distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    gaussian = 1 - np.exp(-(distance**2) / (2 * (D**2)))
    return fshift * gaussian

def ifft(fshift):
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    return iimg

def extract_high_freq_component(img, D):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    high_freq_fshift = gaussian_high_pass_filter(fshift, D)

    high_freq_img = ifft(high_freq_fshift)

    high_freq_img = cv2.normalize(high_freq_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return high_freq_img



def gaussian_filter_high_f(fshift, D):
    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))

    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2

    template = np.exp(- dis_square / (2 * D ** 2))

    return template * fshift


def gaussian_filter_low_f(fshift, D):

    h, w = fshift.shape            # high_freq_part_img = high_freq_part_img.permute()

    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))

    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2

    template = 1 - np.exp(- dis_square / (2 * D ** 2))  # 高斯过滤器

    return template * fshift

def bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



def fft_image(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    return fshift
def ifft(fshift):

    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifftn(ishift)
    iimg = np.abs(iimg)
    return iimg


class HighFrequencyFilter:
    def __init__(self):
        super().__init__()

    def get_low_high_f(self, img, radius_ratio, D):

        f = np.fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform.
        fshift = np.fft.fftshift(f)


        hight_parts_fshift = gaussian_filter_low_f(fshift.copy(), D=D)
        low_parts_fshift = gaussian_filter_high_f(fshift.copy(), D=D)

        low_parts_img = ifft(low_parts_fshift)
        high_parts_img = ifft(hight_parts_fshift)

        img_new_low = (low_parts_img - np.amin(low_parts_img)) / (
                    np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
        img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
                np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)

        # uint8
        img_new_low = np.array(img_new_low * 255, np.uint8)
        img_new_high = np.array(img_new_high * 255, np.uint8)
        return img_new_low, img_new_high



    def apply_filter(self, images, y=1):

            high_freq_results = []
            radius_ratio = 0.5
            for idx, image in enumerate(images):

                img = image.permute(1, 2, 0).numpy()
                image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                D = 70

                low_freq_Part_img, high_freq_part_img = self.get_low_high_f(image, radius_ratio, D)
                if y==0:
                    use_img = low_freq_Part_img
                else:
                    use_img = high_freq_part_img

                fshift1 = fft_image(image)
                fshift2 = fft_image(use_img)
                lam = np.random.beta(1.0, 1.0) * 0.5
                bbx1, bby1, bbx2, bby2 = bbox(fshift2.shape, lam)
                fshift2[bbx1:bbx2, bby1:bby2] = fshift1[bbx1:bbx2, bby1:bby2]
                use_img = ifft(fshift2)

                norm_img = cv2.normalize(use_img, None, 0, 255, cv2.NORM_MINMAX)
                use_img = np.uint8(norm_img)
                high_freq_part_img = cv2.cvtColor(use_img, cv2.COLOR_GRAY2BGR)
                transform = transforms.Compose([transforms.ToTensor()])
                high_freq_part_img = transform(high_freq_part_img)

                high_freq_results.append(high_freq_part_img)

            return high_freq_results

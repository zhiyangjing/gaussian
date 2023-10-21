import cv2
import numpy as np
import matplotlib.pyplot as plt
from pprint import pp
from main import Gaussformula

image = cv2.imread("./sourcedata/boy.jpg")
size = 61
kernel_size = (size, size)
sigma = 5
kernel = cv2.getGaussianKernel(kernel_size[0], sigma) * cv2.getGaussianKernel(kernel_size[1], sigma).T
cortex = Gaussformula(kernel_size[0], sigma)

# 对比两种生成高斯核的方法
print(kernel)
print()
print(cortex)

border_type = cv2.BORDER_REFLECT
border_value = (0, 0, 0)
padding_image = cv2.copyMakeBorder(image, kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2,
                                   kernel_size[1] // 2, border_type, value=border_value)

filtered_image = cv2.filter2D(padding_image, -1, kernel)

height, width, _ = image.shape
cv2.imwrite(f"./output/size{kernel_size[0]}sigma{sigma}cv2.png",
            filtered_image[size // 2:height + size // 2, size // 2:width + size // 2])

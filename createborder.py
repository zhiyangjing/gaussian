import numpy as np
import os
from PIL import Image


def reflectborder(image, bwidth):
    img = np.array(image)
    height, width, _ = img.shape
    resimg = np.zeros((height + 2 * bwidth, width + 2 * bwidth, 3), dtype=np.uint8)
    resimg[bwidth:height + bwidth, bwidth:width + bwidth] = img
    for i in range(bwidth):
        resimg[i, bwidth:width + bwidth] = img[bwidth - i - 1, :]
        resimg[height + bwidth + i, bwidth:width + bwidth] = img[height - i - 1, :]
        resimg[bwidth:height + bwidth, i] = img[:, bwidth - i - 1]
        resimg[bwidth:height + bwidth, width + bwidth + i] = img[:, width - i - 1]
        resimg[:bwidth, i] = img[:bwidth, bwidth - i - 1][::-1]
        resimg[:bwidth, width + bwidth + i] = img[:bwidth, width - i - 1][::-1]
        resimg[height + bwidth:, i] = img[height - bwidth:, bwidth - i - 1][::-1]
        resimg[height + bwidth:, width + bwidth + i] = img[height - bwidth:, width - i - 1][::-1]
    return Image.fromarray(resimg)


def copyedge(image, bwidth):
    img = np.array(image)
    height, width, _ = img.shape
    resimg = np.zeros((height + 2 * bwidth, width + 2 * bwidth, 3), dtype=np.uint8)
    resimg[bwidth:height + bwidth, bwidth:width + bwidth] = img
    for i in range(bwidth):
        resimg[i, bwidth:width + bwidth] = img[0, :]
        resimg[height + bwidth + i, bwidth:width + bwidth] = img[height - 1, :]
        resimg[bwidth:height + bwidth, i] = img[:, 0]
        resimg[bwidth:height + bwidth, width + bwidth + i] = img[:, width - 1]
    resimg[:bwidth, :bwidth] = img[0, 0]
    resimg[:bwidth, width + bwidth:] = img[0, width - 1]
    resimg[height + bwidth:, :bwidth] = img[height - 1, 0]
    resimg[height + bwidth:, width + bwidth:] = img[height - 1, width - 1]
    return Image.fromarray(resimg)


def wraparound(image, bwidth):
    img = np.array(image)
    height, width, _ = img.shape
    resimg = np.zeros((height + 2 * bwidth, width + 2 * bwidth, 3), dtype=np.uint8)
    resimg[bwidth:height + bwidth, bwidth:width + bwidth] = img
    for i in range(bwidth):
        resimg[i, bwidth:width + bwidth] = img[height - bwidth + i, :]
        resimg[height + bwidth + i, bwidth:width + bwidth] = img[i, :]
        resimg[bwidth:height + bwidth, i] = img[:, width - bwidth + i - 1]
        resimg[bwidth:height + bwidth, width + bwidth + i] = img[:,i - 1]
    resimg[:bwidth, :bwidth] = img[height - bwidth:,width - bwidth:]
    resimg[:bwidth, width + bwidth:] = img[height - bwidth:, :bwidth]
    resimg[height + bwidth:, :bwidth] = img[:bwidth:, width - bwidth:]
    resimg[height + bwidth:, width + bwidth:] = img[:bwidth,:bwidth]

    return Image.fromarray(resimg)


if __name__ == "__main__":
    image = Image.open("./sourcedata/boy.jpg")
    if not os.path.exists("./output/border"):
        os.makedirs("./output/border")
    reflectborder(image, 400).save("./output/border/reflectbordertest.png")
    copyedge(image,400).save("./output/border/copyedgetest.png")
    wraparound(image, 400).save("./output/border/wraparoundtest.png")

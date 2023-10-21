from typing import *
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from createborder import reflectborder
from createborder import wraparound
from createborder import copyedge


def Gaussformula(size: int, sigma: float) -> List[List[int]]:
    halfleght = size // 2
    res = [[] for i in range(size)]
    for i in range(-halfleght, halfleght + 1):
        for j in range(-halfleght, halfleght + 1):
            res[i + halfleght].append(1 / (2 * np.pi * sigma ** 2) * np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2)))
    total = sum(sum(res, []))
    for i in range(size):
        for j in range(size):
            res[i][j] /= total
    return np.array(res)


def fileter(cortex, image: Image.Image):
    rchannel: np.ndarray
    gchannel: np.ndarray
    bchannel: np.ndarray
    rchannel, gchannel, bchannel = map(np.array, image.split())
    length = cortex.shape[0]
    halflength = length // 2
    height, width = rchannel.shape
    resimg = np.zeros((height - length + 1, width - length + 1, 3), dtype=np.uint8)
    for i in range(halflength, height - halflength):
        if (i % 100 == 0):
            print(i)
        for j in range(halflength, width - halflength):
            resimg[i - halflength, j - halflength, 0] = np.sum(
                rchannel[i - halflength:i + halflength + 1, j - halflength:j + halflength + 1] * cortex)
            resimg[i - halflength, j - halflength, 1] = np.sum(
                gchannel[i - halflength:i + halflength + 1, j - halflength:j + halflength + 1] * cortex)
            resimg[i - halflength, j - halflength, 2] = np.sum(
                bchannel[i - halflength:i + halflength + 1, j - halflength:j + halflength + 1] * cortex)
    return Image.fromarray(resimg)


if __name__ == "__main__":
    # Size must be odd !
    size = 5
    sigma = 40
    cortex = np.array(Gaussformula(size, sigma))

    x, y = np.meshgrid(np.arange(size), np.arange(size))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, cortex, cmap='coolwarm')
    plt.show()

    # Read image
    image = Image.open('./sourcedata/boy.jpg').convert('RGB')
    print(f"Total height is {image.size[0]},is processing: ")

    # image = reflectborder(image, size // 2)
    image = copyedge(image, size // 2)
    # image = wraparound(image, size // 2)

    resimg = fileter(cortex, image)
    # resimg.save(f"./output/size{size}sigma{sigma}wraparound.png")
    # resimg.save(f"./output/size{size}sigma{sigma}copyedge.png")
    resimg.save(f"./output/size{size}sigma{sigma}reflectborder.png")

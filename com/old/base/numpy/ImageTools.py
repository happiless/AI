import os

from PIL import Image
import numpy as np
import pickle as p


class ImageTools(object):
    image_dir = "./images/"

    result_dir = "./results/"

    data_file_path = "./data.bin"

    def imageToArray(self, files):
        images = np.array([])
        for i in range(len(files)):
            image = Image.open(ImageTools.image_dir + files[i])
            r, g, b = image.split()
            r_array = np.array(r).reshape(19200)
            g_array = np.array(g).reshape(19200)
            b_array = np.array(b).reshape(19200)
            image_array = np.concatenate((r_array, g_array, b_array))
            images = np.concatenate((images, image_array))
            print(images.shape)
            f = open(ImageTools.data_file_path, 'wb')
            p.dump(images, f)
            f.close()

    def arrayToImage(self, file):
        f = open(file, 'rb')
        arr = p.load(f)
        rows = arr.shape[0]
        new_arr = arr.reshape(30, 3, 160, 120)
        for i in range(30):
            r = Image.fromarray(new_arr[i][0]).convert('L')
            g = Image.fromarray(new_arr[i][1]).convert('L')
            b = Image.fromarray(new_arr[i][2]).convert('L')
            image = Image.merge('RGB', (r, g, b))
            image.save(ImageTools.result_dir + str(i) + '.png', 'png')
        print(rows)


if __name__ == '__main__':
    it = ImageTools()
    files = os.listdir(ImageTools.image_dir)
    print(files)
    it.imageToArray(files)
    it.arrayToImage(ImageTools.data_file_path)
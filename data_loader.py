from zipfile import ZipFile
import numpy as np
'''
Name: Kaustubh Hiware
@kaustubhhiware
'''

'''load your data here'''

class DataLoader(object):
    def __init__(self, _bat_size=100):
        self.DIR = './data/'
        self.batch_size = _bat_size
    
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = self.DIR + label_filename + '.zip'
        image_zip = self.DIR + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
            encoded_labels_1hot = []
            for each in labels:
                encoded_labels_1hot.append(np.array([int(i==each) for i in range(10)]))
            
            encoded_labels_1hot = np.array(encoded_labels_1hot)

        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape((len(labels), 28, 28))
        return images, encoded_labels_1hot

    def create_batches(self, x, y, batch_size=100):
        perm = np.random.permutation(len(x))
        x = x[perm]
        y = y[perm]

        batch_x = np.array_split(x, len(x) / batch_size)
        batch_y = np.array_split(y, len(y) / batch_size)

        return batch_x, batch_y

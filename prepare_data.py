from eda import Dataset
import numpy as np
import cv2
import sys
from keras.utils import to_categorical


def print_progress_bar(index, total, label):
    n_bar = 50  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}")
    sys.stdout.flush()


class PrepareData(Dataset):
    def __init__(self,
                list_attr_celeba_path="./data/list_attr_celeba.csv",
                list_bbox_celeba_path="./data/list_bbox_celeba.csv",
                list_eval_partition_path="./data/list_eval_partition.csv",
                list_landmarks_align_celeba_path="./data/list_landmarks_align_celeba.csv",
                images_path="./data/img_align_celeba/"
                ):
        Dataset.__init__(self,
                        list_attr_celeba_path,
                        list_bbox_celeba_path,
                        list_eval_partition_path,
                        list_landmarks_align_celeba_path,
                        images_path)

    def generate_set(self, partition, attribute=['Smiling'], limit=None, color=cv2.COLOR_BGR2RGB):
        """
        partition: 
            0 - train set
            1 - validation set
            2 - test set
        """
        df = self.df.set_index('image_id')

        assert partition in [0,1,2], f"allowed values for partition: {[0,1,2]}"
        data = df.loc[df.partition == partition, attribute]
        if limit:
            data = data.iloc[:limit, :]

        images = np.array(data.index)
        total_images = len(images)
        
        x = []
        for idx, image in enumerate(images, start=1):
            img = cv2.imread(self.images_path + image)
            img = cv2.cvtColor(img, color)
            img = img[20:198, :]  # crop photo to recatangle 178x178
            x.append(img)
            print_progress_bar(idx, total_images, '')
        print()

        x = np.array(x)
        y = np.array(data[attribute]).reshape(-1,)

        return x , self.transform_labels(y)

    @staticmethod
    def transform_labels(labels):
        return np.where(labels < 0, 0, labels)

    @staticmethod
    def save_set(x, y, filename, path='./data/processed/'):
        with open(f'{path+filename}.npy', 'wb') as f:
            np.save(f, x)
            np.save(f, y)

    @staticmethod
    def load_set(filename, path='./data/processed/'):
        with open(f'{path+filename}.npy', 'rb') as f:
            x = np.load(f)
            y = np.load(f)

        return x, y


if __name__ == '__main__':
    dataset = PrepareData()

    x_tr, y_tr = dataset.generate_set(0, limit=10_000)
    dataset.save_set(x_tr, y_tr, 'train')

    x_val, y_val = dataset.generate_set(1, limit=2_000)
    dataset.save_set(x_val, y_val, 'valid')

    x_test, y_test = dataset.generate_set(2, limit=2_000)
    dataset.save_set(x_test, y_test, 'test')

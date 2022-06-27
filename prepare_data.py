from eda import Dataset
import numpy as np
import cv2
import sys

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

    def generate_set(self, partition, attribute=['Smiling'], limit=None):
        """
        partition: 
            0 - train set
            1 - validation set
            2 - test set
        """
        partition_list = [0,1,2]
        assert partition in partition_list, f"allowed values for partition: {partition_list}"

        data = self.df.loc[self.df.partition == partition, attribute]
        if limit:
            data = data.iloc[:limit, :]

        images = np.array(data.index)
        total_images = len(images)
        
        x = []
        for idx, image in enumerate(images, start=1):
            img = cv2.imread(self.images_path + image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x.append(img)
            print_progress_bar(idx, total_images, 'loading images')
        x = np.array(x)

        y = np.array(data[attribute]).reshape(-1,)

        return x , y

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

    # x_tr, y_tr = dataset.generate_set(0)
    # dataset.save_set(x_tr, y_tr, 'train')

    # x_val, y_val = dataset.generate_set(1)
    # dataset.save_set(x_val, y_val, 'valid')




    

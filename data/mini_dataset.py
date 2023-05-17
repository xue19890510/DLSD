from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import pickle
import sys
import importlib,sys
import os.path as osp
importlib.reload(sys)
from PIL import Image
from tqdm import tqdm
import numpy as np

#sys.setdefaultencoding('utf8')
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def load_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo,encoding='bytes')
    return data


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


class miniImageNet_loadfeat(object):
    """
    Dataset statistics:
    # 64 * 600 (train) + 16 * 600 (val) + 20 * 600 (test)
    """
    dataset_dir = '/home/baogui_xu/datasets/MiniImagenet'

    THIS_PATH = osp.dirname(__file__)
    ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
    ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
    IMAGE_PATH1 = osp.join(ROOT_PATH2, '/data00/bgxu/dataset/miniImageNet-only/images')
    SPLIT_PATH = osp.join(ROOT_PATH, '/data00/bgxu/dataset/miniImageNet-only/')

    def __init__(self, **kwargs):
        super(miniImageNet_loadfeat, self).__init__()
        # self.train_dir = os.path.join(self.dataset_dir, 'miniImageNet_category_split_train_phase_train.pickle')
        # self.val_dir = os.path.join(self.dataset_dir, 'miniImageNet_category_split_val.pickle')
        # self.test_dir = os.path.join(self.dataset_dir, 'miniImageNet_category_split_test.pickle')


        train_csv_path = osp.join(miniImageNet_loadfeat.SPLIT_PATH, "train" + '.csv')
        val_csv_path = osp.join(miniImageNet_loadfeat.SPLIT_PATH, "val" + '.csv')
        test_csv_path = osp.join(miniImageNet_loadfeat.SPLIT_PATH, "test" + '.csv')

        load_all_data = False
        if load_all_data:
           self.train, self.train_labels2inds, self.train_labelIds=self.parse_csv_all_data(train_csv_path)
           self.val, self.val_labels2inds, self.val_labelIds = self.parse_csv(val_csv_path)
           self.test, self.test_labels2inds, self.test_labelIds = self.parse_csv_all_data(test_csv_path)
        else:
           self.train, self.train_labels2inds, self.train_labelIds=self.parse_csv(train_csv_path)
           self.val, self.val_labels2inds, self.val_labelIds = self.parse_csv(val_csv_path)
           self.test, self.test_labels2inds, self.test_labelIds = self.parse_csv(test_csv_path)






        # self.train, self.train_labels2inds, self.train_labelIds = self._process_dir(self.train_dir)
        # self.val, self.val_labels2inds, self.val_labelIds = self._process_dir(self.val_dir)
        # self.test, self.test_labels2inds, self.test_labelIds = self._process_dir(self.test_dir)

        self.num_train_cats = len(self.train_labelIds)
        num_total_cats = len(self.train_labelIds) + len(self.val_labelIds) + len(self.test_labelIds)
        num_total_imgs = len(self.train + self.val + self.test)

        print("=> MiniImageNet loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # cats | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(len(self.train_labelIds), len(self.train)))
        print("  val      | {:5d} | {:8d}".format(len(self.val_labelIds),   len(self.val)))
        print("  test     | {:5d} | {:8d}".format(len(self.test_labelIds),  len(self.test)))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_cats, num_total_imgs))
        print("  ------------------------------")

    def parse_csv(self, csv_path):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        lb = -1

        self.wnids = []
        datapair = []
        label2inds = {}
      #  idx=0
        for idx,l in enumerate(tqdm(lines,ncols=64)):
            name, wnid = l.split(',')
            path = osp.join(miniImageNet_loadfeat.IMAGE_PATH1, name)
            if wnid not in self.wnids:

                self.wnids.append(wnid)
                lb += 1
                label2inds[lb]=[]
            label2inds[lb].append(idx)
 #           image = np.array(Image.open(path).convert('RGB'))

            datapair.append((path,lb))

 #           idx = idx+1
        labelIds = sorted(label2inds.keys())

        return datapair,label2inds,labelIds


    def parse_csv_all_data(self, csv_path):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        lb = -1

        self.wnids = []
        datapair = []
        label2inds = {}
      #  idx=0
        for idx,l in enumerate(tqdm(lines,ncols=64)):
            name, wnid = l.split(',')
            path = osp.join(miniImageNet_loadfeat.IMAGE_PATH1, name)
            data = read_image(path)
            if wnid not in self.wnids:

                self.wnids.append(wnid)
                lb += 1
                label2inds[lb]=[]
            label2inds[lb].append(idx)
 #           image = np.array(Image.open(path).convert('RGB'))

            datapair.append((data,lb))

 #           idx = idx+1
        labelIds = sorted(label2inds.keys())

        return datapair,label2inds,labelIds

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _get_pair(self, data, labels):
        assert (data.shape[0] == len(labels))
        data_pair = []
        for i in range(data.shape[0]):
            data_pair.append((data[i], labels[i]))
        return data_pair

    def _process_dir(self, file_path):
        dataset = load_data(file_path)
        data = dataset[b'data']
        print(data.shape)
        labels = dataset[b'labels']
        data_pair = self._get_pair(data, labels)
        labels2inds = buildLabelIndex(labels)
        labelIds = sorted(labels2inds.keys())
        return data_pair, labels2inds, labelIds

if __name__ == '__main__':
    miniImageNet_loadfeat()

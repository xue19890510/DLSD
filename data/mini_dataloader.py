##### from cross attention network https://github.com/blue-blue272/fewshot-CAN
from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader


import torchvision.transforms as transforms
import data.mini_dataset as mini_dataset
import data.mini_sampler_test as sample_test
# import sampler.mini_sampler_train as sample_train
import data.mini_sampler_val as sample_val

import data.mini_sampler_train as sample_train

class DataManager(object):
    """
    Few shot data manager
    """
    def __init__(self,args ):
        super(DataManager, self).__init__()
        self.args = args
        image_size=84
        self.image_size = image_size
        if image_size == 84:
            self.resize_size = 92
        elif image_size == 224:
            self.resize_size = 256


        print("Initializing dataset {}".format("mini"))
        self.normalize_param = dict(mean=[0.472, 0.453, 0.410], std=[0.277, 0.268, 0.285])
        dataset = mini_dataset.miniImageNet_loadfeat()
        transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(self.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.4, 0.4, 0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(**self.normalize_param)
                ])
        transform_test = transforms.Compose([
                    transforms.Resize(self.resize_size),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(**self.normalize_param)
                ])

        # pin_memory = True if use_gpu else False
        pin_memory =False

        self.trainloader = DataLoader(
                sample_train.FewShotDataset_train(name='train_loader',
                    dataset=dataset.train,
                    labels2inds=dataset.train_labels2inds,
                    labelIds=dataset.train_labelIds,
                    nKnovel=5,
                    nExemplars=args.n_shot,
                    nTestNovel=15*5,
                    epoch_size=600,
                    transform=transform_train,
                    load=False,
                ),
                batch_size=1, shuffle=True, num_workers=8,
                pin_memory=pin_memory, drop_last=True,
            )

        self.valloader = DataLoader(
                sample_val.FewShotDataset_val(name='val_loader',
                    dataset=dataset.val,
                    labels2inds=dataset.val_labels2inds,
                    labelIds=dataset.val_labelIds,
                    nKnovel=5,
                    nExemplars=args.n_shot,
                    nTestNovel=15*5,
                    epoch_size=600,
                    transform=transform_test,
                    load=False,
                ),
                batch_size=1, shuffle=False, num_workers=8,
                pin_memory=pin_memory, drop_last=False,
        )
        self.testloader = DataLoader(
               sample_test.FewShotDataset_test(name='test_loader',
                    dataset=dataset.test,
                    labels2inds=dataset.test_labels2inds,
                    labelIds=dataset.test_labelIds,
                    nKnovel=5,
                    nExemplars=args.n_shot,
                    nTestNovel=15*5,
                    epoch_size=2000,
                    transform=transform_test,
                    load=False,
                ),
                batch_size=1, shuffle=False, num_workers=8,
                pin_memory=pin_memory, drop_last=False,
        )

    def return_dataloaders(self):
            return self.trainloader, self.valloader, self.testloader

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchsummary import summary
from ERAV2.main.CIFAR10DataSet import CIFAR10AlbumenationDataSet
from ERAV2.main.Train_Validate import *
from torch_lr_finder import LRFinder


class CIFAR10ResNetUtil:
    def __init__(self, seed=1):
        self.cuda = False
        self.use_mps = False
        self.set_seed(seed)

    def set_seed(self, seed):
        self.cuda = torch.cuda.is_available()
        print("CUDA available: ", self.cuda)
        self.use_mps = torch.backends.mps.is_available()
        print("mps: ", self.use_mps)
        
        if self.cuda:
            torch.cuda.manual_seed(seed)
        elif self.use_mps:
            torch.mps.manual_seed(seed)
        else:
            torch.manual_seed(seed)

    def get_train_transform_cifar10_resnet(self):
        return A.Compose([
            A.PadIfNeeded(36, 36),
            A.RandomCrop(32, 32),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8,
                            fill_value=[0.4914, 0.4822, 0.4465]),
            A.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ToTensorV2()
        ])

    def get_test_transform_cifar10_resnet(self):
        return A.Compose([
            A.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ToTensorV2()
        ])

    def get_train_set_cifar10(self, train_transforms=None):
        if train_transforms is None:
            train_transforms = self.get_train_transform_cifar10_resnet()
        return CIFAR10AlbumenationDataSet('../data', train=True, download=True, transform=train_transforms)

    def get_test_set_cifar10(self, test_transforms=None):
        if test_transforms is None:
            test_transforms = self.get_test_transform_cifar10_resnet()
        return CIFAR10AlbumenationDataSet('../data', train=False, download=True, transform=test_transforms)

    def get_data_loader_args(self):
        dataloader_args = dict(shuffle=True, batch_size=512)
        if self.cuda:
            dataloader_args = dict(shuffle=True, batch_size=512, num_workers=2, pin_memory=True)
        elif self.use_mps:
            dataloader_args = dict(shuffle=True, batch_size=512, pin_memory=True)
        return dataloader_args

    def get_data_loader_cifar10(self, data_set):
        dataloader_args = self.get_data_loader_args()
        return torch.utils.data.DataLoader(data_set, **dataloader_args)

    def get_available_device(self):
        device = torch.device("cpu")
        if self.cuda:
            device = torch.device("cuda")
        elif self.use_mps:
            device = torch.device("mps")
        return device

    def print_summary(self, network, device):
        if self.cuda:
            network.to(device)
        summary(network, input_size=(3, 32, 32))

    def find_lr_fastai(self, model, device, train_loader, criterion, optimizer):
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(train_loader, end_lr=100, num_iter=100, step_mode="exp")
        lr_finder.reset()
        lr_finder.plot()

    def find_lr_leslie_smith(self, model, device, train_loader, test_loader, criterion, optimizer):
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(train_loader, val_loader=test_loader, end_lr=100, num_iter=100, step_mode="exp")
        lr_finder.plot(log_lr=True)
        lr_finder.reset()

    def start_training_testing(self, epochs, collect_images, model, device, train_loader,
                               test_loader, optimizer, criterion, scheduler=None):
        model = model.to(device)
        for epoch in range(epochs):
            print("EPOCH:", epoch)
            print('current Learning Rate: ', optimizer.state_dict()["param_groups"][0]["lr"])
            train(model, device, train_loader, optimizer, epoch, criterion)
            if scheduler is not None:
                scheduler.step()
            test(model, device, test_loader, epoch, epochs, collect_images, criterion)

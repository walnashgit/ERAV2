import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ERAV2.main.CIFAR10DataSet import CIFAR10AlbumenationDataSet
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from ERAV2.main.Train_Validate import *
try:
    from torch_lr_finder import LRFinder
except ImportError:
    # Run from source
    import sys

    sys.path.insert(0, '..')
    from torch_lr_finder import LRFinder



SEED = 1
cuda = False
use_mps = False


def set_seed(seed = 1):
  SEED = seed
  # for CUDA
  cuda = torch.cuda.is_available()
  print("CUDA available: ", cuda)

  # for Apple GPU
  use_mps = torch.backends.mps.is_available()
  print("mps: ", use_mps)

  if cuda:
      torch.cuda.manual_seed(SEED)
  elif use_mps:
      torch.mps.manual_seed(SEED)
  else:
      torch.manual_seed(SEED)

    
def get_train_transform_cifar10_resnet():
    return A.Compose([
        A.PadIfNeeded(36, 36),
        A.RandomCrop(32, 32),
        A.HorizontalFlip(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=25, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8,
                        fill_value=[0.4914, 0.4822, 0.4465]),
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ToTensorV2()
    ])


def get_test_transform_cifar10_resnet():
    return A.Compose([
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ToTensorV2()
    ])


def get_train_set_cifar10(train_transforms = get_train_transform_cifar10_resnet()):
    return CIFAR10AlbumenationDataSet('../data', train=True, download=True, transform=train_transforms)
   

def get_test_set_cifar10(test_transforms = get_test_transform_cifar10_resnet()):
    return CIFAR10AlbumenationDataSet('../data', train=False, download=True, transform=test_transforms)


def get_data_loader_args():
  dataloader_args = dict(shuffle=True, batch_size=512)

  if cuda:
      dataloader_args = dict(shuffle=True, batch_size=512, num_workers=2, pin_memory=True)
  elif use_mps:
      dataloader_args = dict(shuffle=True, batch_size=512, pin_memory=True)

  return dataloader_args


def get_data_loader_cifar10(data_set):
  dataloader_args = get_data_loader_args()
  return torch.utils.data.DataLoader(data_set, **dataloader_args)


def get_available_device():
    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    return device


def print_summary(network, device):
    if cuda:
        network.to(device)
    summary(network, input_size=(3, 32, 32))


def find_lr_fastai(model, device, train_loader, criterion, optimizer):
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-10, weight_decay=1e-2)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100, step_mode="exp")
    # lr_finder.plot()
    lr_finder.reset()
    lr_finder.plot()


def find_lr_leslie_smith(model, device, train_loader, test_loader, criterion, optimizer):
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-2)
    # optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, val_loader=test_loader, end_lr=100, num_iter=100, step_mode="exp")
    lr_finder.plot(log_lr=True)
    lr_finder.reset()


def start_training_testing(epochs, collect_images, model, device, train_loader, 
  test_loader, optimizer, criterion, scheduler=None):
    # Running the model
    model = model.to(device)
    # L2 regularization
    # optim_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0)  # using L2 regularization
    # optim_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # optim_adam = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)  # using L2 regularization
    # optim_adam = optim.Adam(model.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.StepLR(optim_sgd, step_size=5, gamma=0.1, verbose=True)
    # scheduler = optim.lr_scheduler.CyclicLR(optim_adam, base_lr=0.000572, max_lr=0.00572, step_size_up=5, step_size_down=19, mode="triangular", cycle_momentum=False)

    for epoch in range(epochs):
        print("EPOCH:", epoch)
        print('current Learning Rate: ', optimizer.state_dict()["param_groups"][0]["lr"])
        train(model, device, train_loader, optimizer, epoch, criterion)

        if scheduler != None:
          scheduler.step()
        test(model, device, test_loader, epoch, epochs, collect_images, criterion)

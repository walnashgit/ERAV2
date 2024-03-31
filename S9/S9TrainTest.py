from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from S9DilatedConvModel import Network
from torchsummary import summary
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Util import Cifar10TrainDataset

# Check the data shape, mean and std deviation
# simple_transforms = transforms.Compose([
#     transforms.ToTensor()
# ])

# exp = datasets.CIFAR10('./data', train = True, download = True, transform = simple_transforms)
#
# data = exp.data / 255  # dividing by max pixel value to scale the data between 0-1
# print(' - data.mean():', data.mean(axis=(0,1,2)))
# print(' - data.std():', data.std(axis=(0,1,2)))

# - data.mean(): [0.49139968 0.48215841 0.44653091]
# - data.std(): [0.24703223 0.24348513 0.26158784]

#Prepare transforms using mean and std from above.
train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=25, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16,
                        fill_value=[0.4914, 0.4822, 0.4465]),
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ToTensorV2()
])

test_transforms = A.Compose([
    A.Normalize((0.4914, 0.4822, 0.4465),(0.2471, 0.2435, 0.2616)),
    ToTensorV2()
])

# train_set = datasets.CIFAR10('../data', train = True, download = True, transform = train_transforms)
train_set = Cifar10TrainDataset('../data', train = True, download = True, transform = train_transforms)
test_set = Cifar10TrainDataset('../data', train = False, download = True, transform = test_transforms)


# verifying that mean and std deviation values used are correct
# transformed_data = []
# for image in train.data:
#     transformed_image = train.transform(image)
#     transformed_data.append(transformed_image)
#
# transformed_data = torch.stack(transformed_data)
# print(' - transformed_data.shape:', transformed_data.shape)
# print(' - transformed_data.mean():', transformed_data.mean(axis=(0,2,3)))
# print(' - transformed_data.std():', transformed_data.std(axis=(0,2,3)))

SEED = 4
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

# Create data loader, default for CPU
dataloader_args = dict(shuffle=True, batch_size=128)

if cuda:
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True)
elif use_mps:
    dataloader_args = dict(shuffle=True, batch_size=128, pin_memory=True)

# Train loader
train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)

# Test loader
test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)

# Select device based on available GPU/CPU
device = torch.device("cpu")
if cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")


# Print summary. Summary is printed using CPU as torchsummary does not support APPlE GPU
network = Network()

if cuda:
    network.to(device)
summary(network, input_size=(3, 32, 32))

from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []
missclassified_images = []


def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    #print('data.shape: ', data.shape)
    y_pred = model(data)

    # print('- y_pred:', y_pred.shape)

    # Calculate loss
    #loss = F.cross_entropy(y_pred, target) # nn.CrossEntropyLoss(y_pred, target)
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)


def test(model, device, test_loader, epoch, EPOCHS, collect_images):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print('- output.shape:', output.shape) # torch.Size([128, 10])
            # test_loss += nn.CrossEntropyLoss(output, target, reduction='sum')
            #test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # find miss-classified images in last epoch
            if collect_images and epoch == EPOCHS - 1:
                misclassified_indices = (pred != target.view_as(pred)).nonzero()[:, 0]
                for idx in misclassified_indices:
                    missclassified_images.append((data[idx].cpu(), target[idx].cpu(), pred[idx].cpu()))


    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))


def plot_misclassified_images(misclassified_images, num_images=10):
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(wspace=0.8, hspace=0.8)
    for i in range(min(num_images, len(misclassified_images))):
        image, true_label, predicted_label = misclassified_images[i]
        ax = fig.add_subplot(5, 2, i + 1)
        ax.imshow(np.transpose(image, (1, 2, 0)))
        ax.set_title(f'True: {train_set.classes[true_label]}, \n Predicted: {train_set.classes[predicted_label]}')
        ax.axis('off')
    plt.show()


def start_training_testing(epochs, collect_images):
    # Running the model
    model = network.to(device)
    # L2 regularization
    # optim_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0)  # using L2 regularization
    optim_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # optim_adam = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)  # using L2 regularization
    # scheduler = optim.lr_scheduler.StepLR(optim_sgd, step_size=5, gamma=0.1, verbose=True)

    for epoch in range(epochs):
        print("EPOCH:", epoch)
        train(model, device, train_loader, optim_sgd, epoch)
        test(model, device, test_loader, epoch, epochs, collect_images)
        # scheduler.step()


EPOCHS = 20
start_training_testing(EPOCHS, False)

# plot_misclassified_images(missclassified_images)



# visualize data
# batch = next(iter(train_loader))
# images, labels = batch
#
# plt.figure(figsize=(1.5,1.5))
# image = np.transpose(images[4], (1,2,0))
# plt.imshow(image)
# plt.title(train_set.classes[labels[4]])
# plt.show()

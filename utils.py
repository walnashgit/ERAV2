import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary


# Returns train data transformations
def getTrainingTransforms():
    return transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

# Returns test data transformations
def getTestTransforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

# Downloads and returns the training data.
def getTrainingData(train_transforms):
    return datasets.MNIST('../data', train=True, download=True, transform=train_transforms)


# Downloads and returns the test data.
def getTestData(test_transforms):
    return datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

# Returns a dataLoader created with the data and the batch size passed.
def getDataLoader(data, batch_size):
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
    return torch.utils.data.DataLoader(data, **kwargs)


# Plots the training data sample for visualisation. Pass the training data as dataLoader and the sample_size to visualize
def plotTrainDataSample(train_loader, sample_size):
    batch_data, batch_label = next(iter(train_loader))

    fig = plt.figure()

    for i in range(sample_size):
      plt.subplot(3,4,i+1)
      plt.tight_layout()
      plt.imshow(batch_data[i].squeeze(0), cmap='gray')
      plt.title(batch_label[i].item())
      plt.xticks([])
      plt.yticks([])


# Returns an optimizer
def getOptimizer(model):
    return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# Returns a scheduler
def getScheduler(optimizer):
    return optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)


# Returns a criterion
def getCriterion():
    return nn.CrossEntropyLoss()

# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

# Returns the correct prediction count.
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


# Trains the model passed with the training data passed as train_loader.
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

# Test the model that was trained against the training data passed as test_loader.
def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            #test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, target).item()

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Plots the graph for training's and test's loss and accuracy data collected during training and test.
def plotTrainingTestLossAndAccuracy():
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


# Retruns the available device
def getAvailableDevice():
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


# Prints the summary of the model. Requires torchsummary to be installed.
def printModelSummary(model):
    summary(model, input_size=(1, 28, 28))

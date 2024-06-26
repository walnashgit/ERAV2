import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import numpy as np

train_losses = []
test_losses = []
train_acc = []
test_acc = []
missclassified_images = []


def train(model, device, train_loader, optimizer, epoch, criterion):
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
        y_pred = model(data)

        # Calculate loss
        # loss = F.cross_entropy(y_pred, target)

        # criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, target)

        # loss = F.nll_loss(y_pred, target)
        train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

        train_acc.append(100 * correct / processed)


def test(model, device, test_loader, epoch, EPOCHS, collect_images, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print('- output.shape:', output.shape) # torch.Size([128, 10])

            test_loss += criterion(output, target).item()

            # test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss

            # test_loss += F.nll_loss(output, target, reduction='sum').item()

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


def plot_misclassified_images(misclassified_images, train_set, num_images=10):
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(wspace=0.8, hspace=0.8)
    for i in range(min(num_images, len(misclassified_images))):
        image, true_label, predicted_label = misclassified_images[i]
        ax = fig.add_subplot(5, 2, i + 1)
        ax.imshow(np.transpose(image, (1, 2, 0)))
        ax.set_title(f'True: {train_set.classes[true_label]}, \n Predicted: {train_set.classes[predicted_label]}')
        ax.axis('off')
    plt.show()

import cv2
import torchvision

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Cifar10TrainDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root: str, train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

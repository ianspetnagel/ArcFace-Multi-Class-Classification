import matplotlib.pyplot as plt
import torchvision as tv
import torchvision
from torch.utils.data import DataLoader,Dataset
import torch

def get_loaders_MNIST(batch_size=100):
    transforms =tv.transforms.Compose([tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.1307,), (0.3081,))])
    train_data = tv.datasets.MNIST( root="./data/",train=True,download=True,transform = transforms)
    test_data = tv.datasets.MNIST( root="./data/",train=False,download=True,transform = transforms)
    train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True,drop_last=True,num_workers=2)
    test_loader = DataLoader(dataset = test_data,batch_size = batch_size,shuffle = False)
    return train_loader, test_loader

def get_loaders_CIFAR10(batch_size=100):
    transform_train = tv.transforms.Compose([tv.transforms.RandomCrop(32, padding=4),tv.transforms.RandomHorizontalFlip(),tv.transforms.ToTensor(),tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,0.2010)),])
    transform_test = tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,0.2010)),])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform= transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    return train_loader, test_loader

def decet(feature,targets,epoch,save_path):
    color = ["red", "black", "yellow", "green", "pink","gray", "lightgreen", "orange", "blue", "teal"]
    cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.ion()
    plt.clf()
    for j in cls:
        mask = [targets == j]
        feature_ = feature[mask].numpy()
        x = feature_[:, 1]
        y = feature_[:, 0]
        label = cls
        plt.plot(x, y, ".", color=color[j])
        plt.legend(label, loc="upper right")
        plt.title("epoch={}".format(str(epoch+1)))
    plt.savefig('{}/{}.jpg'.format(save_path,epoch+1))
    plt.draw()
    plt.pause(0.01)


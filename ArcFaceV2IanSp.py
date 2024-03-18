import sys
from Network import Network
from ArcNet import ArcNet
import utils
import torch
import torch.nn as nn
import torch as t
#from utils import decet

#import os

## create directory if it does not exist
#if not os.path.exists(save_path):
#    os.makedirs(save_path)


def testdata_accuracy(CUDA, net, arcnet,test_loader):
    acc = 0
    for i, (x, y) in enumerate(test_loader):
        if CUDA:
            x = x.cuda()
            y = y.cuda()
        latent_out = net(x)
        arc_out = t.log(arcnet(latent_out))
        #---------test accuracy-----------------
        value = t.argmax(arc_out, dim=1)
        acc += t.sum((value == y).float())

    print('test accuracy = ', acc / len(y))

def main():
    latent_dim = 3 # embedding size
    num_classes = 10
    net = Network(latent_dim)
    #net = ResNet50(10, latent_dim, channels=3)
    CUDA = torch.cuda.is_available()
    if CUDA:
        net = net.cuda()
    arcnet = ArcNet(num_classes, latent_dim).cuda()
    arcloss = nn.NLLLoss(reduction="mean").cuda()
    optimizerarc = t.optim.SGD([{'params': net.parameters()}, {'params': arcnet.parameters()}], lr=0.01, momentum=0.9, weight_decay=0.0005)
    save_pic_path = "./Images"
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    num_epochs = 80
    train_loader, test_loader = utils.get_loaders_CIFAR10()
    #train_loader, test_loader = utils.get_loaders_MNIST()

    for epoch in range(num_epochs):
        correct = 0
        iterations = 0
        iter_loss = 0.0
        feat = []
        target = []
        net.train()
        embedding = []
        target = []
        for i, (x, y) in enumerate(train_loader):
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            latent_out = net(x)
            #-------------loss calculation--------------
            arc_out = t.log(arcnet(latent_out))
            arcface_loss = arcloss(arc_out, y)
            iter_loss = arcface_loss.item()
            #-------------compute accuracy-------------
            value = t.argmax(arc_out, dim=1)
            acc = t.sum((value == y).float()) / len(y)
            #----------compute gradients, update network parameters----
            optimizerarc.zero_grad()
            arcface_loss.backward()
            optimizerarc.step()
            embedding.append(latent_out)
            target.append(y)

            iterations += 1
            if epoch > 10:
                optimizerarc.param_groups[0]['lr'] = 0.005
            if epoch > 20:
                optimizerarc.param_groups[0]['lr'] = 0.002
            if epoch > 40:
                optimizerarc.param_groups[0]['lr'] = 0.001
            if epoch > 60:
                optimizerarc.param_groups[0]['lr'] = 0.0001

        features = t.cat(embedding, 0)
        targets = t.cat(target, 0)
        utils.decet(features.data.cpu(), targets.data.cpu(),epoch,save_pic_path)
        train_loss.append(iter_loss/iterations)
        train_accuracy.append(acc)
        print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}'
        .format(epoch+1, num_epochs, train_loss[-1], train_accuracy[-1]))

        testdata_accuracy(CUDA, net,arcnet, test_loader)

    PATH = "./trainedmodel/model.pth"
    torch.save(net.state_dict(),PATH)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
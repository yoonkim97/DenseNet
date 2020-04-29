import time
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms
import numpy as np
from densenet import DenseNet3
import torch.nn as nn
from torch import optim

train_label_classes = [2]
test_label_classes = [1]

batch_size = 64
validation_ratio = 0.1
random_seed = 10
initial_lr = 0.1
num_epoch = 50

def get_same_indices(target, labels):
  label_indices = []
  for i in range (len(target)):
    for j in range (len(labels)):
      if target[i] == labels[j]:
        label_indices.append(i)
  return label_indices

def get_dataloaders():

    transform_train = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
    transform_validation = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])

    traindir = '/home/yoon/jyk416/OneClassDenseNet/data'
    testdir = '/home/yoon/jyk416/OneClassDenseNet/data'

    trainset = torchvision.datasets.ImageFolder(root=traindir, transform=transform_train)
    validset = torchvision.datasets.ImageFolder(root=traindir, transform=transform_validation)
    testset = torchvision.datasets.ImageFolder(root=testdir, transform=transform_test)

    test_targets = []
    train_targets = []
    for target in trainset.targets:
        if target == 0:
            test_targets.append(target)
        else:
            train_targets.append(target)

    print(test_targets, len(test_targets))
    print(train_targets, len(train_targets))

    num_train = len(trainset)
    indices = get_same_indices(trainset.targets, train_label_classes)
    split = int(np.floor(validation_ratio * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    test_indices = get_same_indices(testset.targets, test_label_classes)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler, num_workers=0
    )

    valid_loader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, sampler=valid_sampler, num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=0
    )

    return train_loader, valid_loader, test_loader

def DenseNetBC_100_12():
    return DenseNet3(depth=100, num_classes=1, growth_rate=12, reduction=0.5, bottleneck=True, dropRate=0.2)

def main():
    train_loader, valid_loader, test_loader = get_dataloaders()

#     start_ts = time.time()
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     model = DenseNetBC_100_12().to(device)
#
#     losses = []
#     loss_function = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
#     lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
#                                                   milestones=[int(num_epoch * 0.5), int(num_epoch * 0.75)], gamma=0.1,
#                                                   last_epoch=-1)
#     # training loop + validation loop
#     for epoch in range(num_epoch):
#         lr_scheduler.step()
#         total_loss = 0.0
#
#         for i, data in enumerate(train_loader, 0):
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             model.zero_grad()
#             outputs = model(inputs)
#             loss = loss_function(outputs, labels)
#
#             loss.backward()
#             optimizer.step()
#             current_loss = loss.item()
#             total_loss += current_loss
#
#             show_period = 100
#             print('[%d, %d/50000] loss: %.7f' % (epoch + 1, (i + 1) * batch_size, total_loss / show_period))
#             total_loss = 0.0
#     torch.cuda.empty_cache()
#
#     # validation part
#     correct = 0
#     total = 0
#
#     for i, data in enumerate(valid_loader, 0):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#         print('[%d epoch] Accuracy of the network on the validation images: %d %%' %
#               (epoch + 1, 100 * correct / total))
#
# print('Finished Training')

if __name__ == '__main__':
    main()
import time
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms
import numpy as np
from densenet import DenseNet3
import torch.nn as nn
from torch import optim

female_label_class = [0]
male_label_class = [1]

batch_size = 4
validation_ratio = 0.1
random_seed = 10
initial_lr = 0.1
num_epoch = 50


def get_same_indices(target, labels):
    label_indices = []
    for i in range(len(target)):
        for j in range(len(labels)):
            if target[i] == labels[j]:
                label_indices.append(i)
    return label_indices


def get_dataloaders():
    transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
    # transform_train = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
    # transform_validation = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
    # transform_test = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])

    train_test_dir = '/home/yoon/jyk416/OneClassDenseNet/data/train'
    # traindir = '/home/yoon/jyk416/OneClassDenseNet/data'
    # testdir = '/home/yoon/jyk416/OneClassDenseNet/data'

    dataset = torchvision.datasets.ImageFolder(root=train_test_dir, transform=transform)
    # trainset = torchvision.datasets.ImageFolder(root=traindir, transform=transform_train)
    # validset = torchvision.datasets.ImageFolder(root=traindir, transform=transform_validation)
    # testset = torchvision.datasets.ImageFolder(root=testdir, transform=transform_test)

    female_count = 0
    male_count = 0
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == 0:
            female_count += 1
        else:
            male_count += 1

    num_female_train = female_count
    num_male_train = male_count
    female_indices = get_same_indices(dataset.targets, female_label_class)
    male_indices = get_same_indices(dataset.targets, male_label_class)
    split_female = int(np.floor(validation_ratio * num_female_train))
    split_male = int(np.floor(validation_ratio * num_male_train))

    num_train = len(dataset)
    indices = get_same_indices(dataset.targets, female_label_class)
    split = int(np.floor(validation_ratio * num_train))

    female_train_idx, female_valid_idx = female_indices[split_female:], female_indices[:split_female]
    male_train_idx, male_valid_idx = male_indices[split_male:], male_indices[split_male:]
    print("sum: ", len(male_train_idx) + len(male_valid_idx))
    female_train = female_train_idx + male_train_idx
    print("female training", female_train, len(female_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    test_indices = get_same_indices(dataset.targets, male_label_class)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=0
    )

    return train_loader, valid_loader, test_loader


def DenseNetBC_100_12():
    return DenseNet3(depth=100, num_classes=1, growth_rate=12, reduction=0.5, bottleneck=True, dropRate=0.2)


def train():
    train_loader, valid_loader, test_loader = get_dataloaders()

    start_ts = time.time()
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DenseNetBC_100_12().to(device)
    losses = []
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                  milestones=[int(num_epoch * 0.5), int(num_epoch * 0.75)], gamma=0.1,
                                                  last_epoch=-1)
    # training loop + validation loop
    for epoch in range(num_epoch):
        lr_scheduler.step()
        total_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            for i in range(len(labels)):
                labels[i] = 0
            model.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
            show_period = 100
            print('[%d, %d/50000] loss: %.7f' % (epoch + 1, (i + 1) * batch_size, total_loss / show_period))
            total_loss = 0.0
    torch.cuda.empty_cache()

    # validation part
    correct = 0
    total = 0

    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        for i in range(len(labels)):
            labels[i] = 0
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        print('[%d epoch] Accuracy of the network on the validation images: %d %%' %
              (epoch + 1, 100 * correct / total))

    print('Finished Training')


def main():
    train()


if __name__ == '__main__':
    main()

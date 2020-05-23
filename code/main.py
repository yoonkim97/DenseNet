import time
import os
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms
import numpy as np
from densenet import DenseNet3
import torch.nn as nn
from torch import optim

# female_label_class = [0]
# male_label_class = [1]

# ap_label_class = [0]
# pa_label_class = [1]

healthynocardiomegaly_label_class = [0]
unhealthynocardiomegaly_label_class = [1]

batch_size = 8
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

    train_test_dir = '/home/yoon/jyk416/OneClassDenseNet/data/train3'
    # traindir = '/home/yoon/jyk416/OneClassDenseNet/data'
    # testdir = '/home/yoon/jyk416/OneClassDenseNet/data'

    dataset = torchvision.datasets.ImageFolder(root=train_test_dir, transform=transform)
    # trainset = torchvision.datasets.ImageFolder(root=traindir, transform=transform_train)
    # validset = torchvision.datasets.ImageFolder(root=traindir, transform=transform_validation)
    # testset = torchvision.datasets.ImageFolder(root=testdir, transform=transform_test)

    # female_count = 0
    # male_count = 0
    # for i in range(len(dataset.targets)):
    #     if dataset.targets[i] == 0:
    #         female_count += 1
    #     else:
    #         male_count += 1
    #
    # num_female_train = female_count
    # num_male_train = male_count
    # female_indices = get_same_indices(dataset.targets, female_label_class)
    # male_indices = get_same_indices(dataset.targets, male_label_class)
    # split_female = int(np.floor(validation_ratio * num_female_train))
    # split_male = int(np.floor(validation_ratio * num_male_train))

    # ap_count = 0
    # pa_count = 0
    # for i in range(len(dataset.targets)):
    #     if dataset.targets[i] == 0:
    #         ap_count += 1
    #     else:
    #         pa_count += 1
    #
    # num_ap_train = ap_count
    # num_pa_train = pa_count
    # ap_indices = get_same_indices(dataset.targets, ap_label_class)
    # pa_indices = get_same_indices(dataset.targets, pa_label_class)
    # split_ap = int(np.floor(validation_ratio * num_ap_train))
    # split_pa = int(np.floor(validation_ratio * num_pa_train))

    healthynocardiomegaly_count = 0
    unhealthynocardiomegaly_count = 0
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == 0:
            healthynocardiomegaly_count += 1
        else:
            unhealthynocardiomegaly_count += 1
    num_healthynocardiomegaly_train = healthynocardiomegaly_count
    num_unhealthynocardiomegaly_train = unhealthynocardiomegaly_count
    healthynocardiomegaly_indices = get_same_indices(dataset.targets, healthynocardiomegaly_label_class)
    unhealthynocardiomegaly_indices = get_same_indices(dataset.targets, unhealthynocardiomegaly_label_class)
    split_healthynocardiomegaly = int(np.floor(validation_ratio * num_healthynocardiomegaly_train))
    split_unhealthynocardiomegaly = int(np.floor(validation_ratio * num_unhealthynocardiomegaly_train))

    # num_train = len(dataset)
    # indices = get_same_indices(dataset.targets, female_label_class)
    # split = int(np.floor(validation_ratio * num_train))

    # female_train_idx, female_valid_idx = female_indices[split_female:], female_indices[:split_female]
    # male_train_idx, male_valid_idx = male_indices[split_male:], male_indices[:split_male]
    # train_set = female_train_idx + male_train_idx
    # valid_set = female_valid_idx + male_valid_idx

    # ap_train_idx, ap_valid_idx = ap_indices[split_ap:], ap_indices[:split_ap]
    # pa_train_idx, pa_valid_idx = pa_indices[split_pa:], pa_indices[:split_pa]
    # train_set = ap_train_idx + pa_train_idx
    # valid_set = ap_valid_idx + pa_valid_idx

    healthynocardiomegaly_train_idx, healthynocardiomegaly_valid_idx = healthynocardiomegaly_indices[split_healthynocardiomegaly:], healthynocardiomegaly_indices[:split_healthynocardiomegaly]
    unhealthynocardiomegaly_train_idx, unhealthynocardiomegaly_valid_idx = unhealthynocardiomegaly_indices[split_unhealthynocardiomegaly:], unhealthynocardiomegaly_indices[:split_unhealthynocardiomegaly]
    train_set = healthynocardiomegaly_train_idx + unhealthynocardiomegaly_train_idx
    valid_set = healthynocardiomegaly_valid_idx + unhealthynocardiomegaly_valid_idx

    # train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_set)
    valid_sampler = SubsetRandomSampler(valid_set)

    # test_indices = get_same_indices(dataset.targets, male_label_class)
    # test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0
    )

    # test_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=0
    # )

    return train_loader, valid_loader


def DenseNetBC_50_12():
    return DenseNet3(depth=50, num_classes=2, growth_rate=12, reduction=0.5, bottleneck=True, dropRate=0.2)

def save_checkpoint(state, is_best, filename='/home/yoon/jyk416/OneClassDenseNet/checkpoints/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")

def train():
    directory = '/home/yoon/jyk416/OneClassDenseNet/checkpoints/'
    model_directory = '/home/yoon/jyk416/OneClassDenseNet/models/'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    train_loader, valid_loader = get_dataloaders()

    start_ts = time.time()
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNetBC_50_12().to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                  milestones=[int(num_epoch * 0.5), int(num_epoch * 0.75)], gamma=0.1,
                                                  last_epoch=-1)
    best_accuracy = 0
    resume_weights = "/home/yoon/jyk416/OneClassDenseNet/checkpoints/checkpoint.pth.tar"
    start_epoch = 0

    if os.path.exists(resume_weights):
        # cuda = torch.cuda.is_available()
        if torch.cuda.is_available():
            checkpoint = torch.load(resume_weights)
        else:
            # Load GPU model on CPU
            checkpoint = torch.load(resume_weights,
                                    map_location=lambda storage,
                                                        loc: storage)
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))

    model_filename = '/home/yoon/jyk416/OneClassDenseNet/models/model{}.pth'
    # training loop + validation loop
    for epoch in range(num_epoch):
        lr_scheduler.step()
        total_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            show_period = 100
            print('[%d, %d/76336] loss: %.7f' % (start_epoch + epoch + 1, (i + 1) * batch_size, total_loss / show_period))
            total_loss = 0.0
        torch.cuda.empty_cache()

        # validation part
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(valid_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                print('[%d epoch] Accuracy of the network on the validation images: %d %%' %
                      (start_epoch + epoch + 1, accuracy))

                is_best = bool(accuracy > best_accuracy)
                best_accuracy = max(accuracy, best_accuracy)

                if not os.path.exists(directory):
                    os.makedirs(directory)

                save_checkpoint({
                    'epoch': start_epoch + epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_accuracy': best_accuracy
                }, is_best)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), model_filename.format(epoch + 1))
    torch.cuda.empty_cache()
    print('Finished Training')


def main():
    train()


if __name__ == '__main__':
    main()

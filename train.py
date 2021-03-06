# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import time 
import sys
import torch
import cv2
import os
import argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import heapq

from tensorboardX import SummaryWriter
from torchvision import transforms
from models.mobilenetv3 import mobilenetv3
from dataloader import MyDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device available: {}'.format(device))

def format_second(secs):                                      
    h = int(secs / 3600)
    m = int((secs % 3600) / 60)
    s = int(secs % 60)
    ss = "Exa(h:m:s):{:0>2}:{:0>2}:{:0>2}".format(h,m,s)
    return ss    



def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--train_txt', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--val_txt', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--gpus', help='Use CUDA on the listed devides', nargs='+', type=int, default=[])
    parser.add_argument('--seed', help='Random seed', type=int, default=1234)
    parser.add_argument('--batch_size',  help='Batch size', type=int, default=128)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--start_epoch', help='start epoch', type=int, default=1)
    parser.add_argument('--input_size', help='Image size', type=int, default=60)

    parser.add_argument('--model_name', help='name of the model to save')
    parser.add_argument('--pretrained', help='pretrained model name')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def main(args=None):
    train_transform = transforms.Compose([
        transforms.Resize(60),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(60),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset_train = MyDataset(pathDataset=args.train_txt, transform=train_transform)
    dataset_val = MyDataset(pathDataset=args.val_txt, transform=test_transform)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False)

    batch_num = len(dataset_train) // args.batch_size

    model = mobilenetv3()
    print('network:')
    print(model)

    save_path = './checkpoint'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    use_cuda = torch.cuda.is_available() and len(args.gpus) > 0
    if use_cuda:
        torch.cuda.set_device(args.gpus[0])
        torch.cuda.manual_seed(args.seed)
        #model = torch.nn.DataParallel(model, device_ids=args.gpus)
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)

    milestones = [50, 80, 120, 150]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    writer = SummaryWriter(log_dir='./summary')

    for epoch in range(args.start_epoch, args.epochs+1):
        train_loss = train(dataloader_train, model, criterion, optimizer, epoch, scheduler, batch_num, args.epochs)
        test_loss = test(dataloader_val, model, criterion)

        scheduler.step()

        model_name = 'Mask_Glass_Classify'
        if epoch % 2 == 0:

            save_name = '{}/{}_{}.pth.tar'.format(save_path, model_name, epoch)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, filename=save_name)


        
        writer.add_scalars('scalar/loss', {'train_loss': train_loss, 'test_loss': test_loss}, epoch)

    writer.export_scalars_to_json('./summary/' + 'pretrain' + 'all_scalars.json')
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, scheduler, batch_number , epochs):
    model.train()

    iter_samples = 0
    correct = 0
    running_loss = 0.0
    count_loss = []

    t0 = time.time()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = Variable(inputs).float().to(device)
        labels = Variable(labels).long().to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        softmax_output = torch.softmax(outputs, dim=-1)
        _, predicted = torch.max(softmax_output.data, 1)

        running_loss += loss.item()
        count_loss.append(float(loss))

        iter_samples += labels.size(0)

        correct += (predicted==labels).sum().item()

        if (i+1)%20 == 0 or (i+1) == batch_number:
            t1 = time.time()
            speed = (t1 - t0) / i 
            exp_time = format_second(speed * (len(train_loader) * (epochs - epoch + 1) - i))
            learning_rate = scheduler.get_last_lr()[0]
            print('[epoch: {} | iter: {}/{}] | loss: {:1.5f} | acc: {:1.5f} | lr: {} | exp_time: {}'.format(

                epoch, (i+1), batch_number, (running_loss/20), (100.0*correct/iter_samples), learning_rate, exp_time))
            running_loss = 0.0
            correct = 0
            iter_samples = 0

    mean_loss = np.mean(count_loss)
    return mean_loss


def test(test_loader, model, criterion):
    model.eval()

    total = 0
    correct = 0
    running_loss = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels= data
            inputs = Variable(inputs).float().to(device)
            labels = Variable(labels).long().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss.append(float(loss))

            softmax_output = torch.softmax(outputs, dim=-1)
            _, predicted = torch.max(softmax_output.data, 1)

            total += len(labels)
            correct += (predicted==labels).sum().item()

    test_loss = np.mean(running_loss)
    print('test loss:{:1.5f} | test acc:{:1.5f}%'.format(test_loss, (100.0*correct/total)))
    return test_loss

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print('save model: {}\n'.format(filename))
    torch.save(state, filename)


if __name__ == '__main__':
    args = parse_args()
    main(args)



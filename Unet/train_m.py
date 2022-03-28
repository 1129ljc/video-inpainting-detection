import os
import cv2
import torch
import random
import argparse
import datetime
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from model import VideoInpaintingDetectionNet
from loss import FocalLoss
from utils import F1, IOU
from torch.utils.data import Dataset
from torchvision import transforms


class IID_Dataset(Dataset):
    def __init__(self, choice='train'):
        self.input_size = (512, 512)
        if choice == 'train':
            self.image_path = '../dataset/DSTT-quality/1/train'
            self.mask_path = '../dataset/mask_white/'
        if choice == 'test':
            self.image_path = '../dataset/DSTT-quality/1/val'
            self.mask_path = '../dataset/mask_white/'
        self.train_files = []
        self.choice = choice
        names = os.listdir(self.image_path)
        for i in range(len(names)):
            name = names[i]
            image_name_dir = os.path.join(self.image_path, name)
            mask_name_dir = os.path.join(self.mask_path, name)
            mask_files = sorted(os.listdir(mask_name_dir))
            mask_num = len(mask_files)
            image_files = sorted(os.listdir(image_name_dir))
            image_num = len(image_files)
            frame_num = min(mask_num, image_num)
            for j in range(frame_num):
                mask_file = os.path.join(mask_name_dir, mask_files[j])
                image_file = os.path.join(image_name_dir, image_files[j])
                self.train_files.append([image_file, mask_file])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def tensor(self, img):
        img = img / 255.
        return torch.from_numpy(img).float().permute(2, 0, 1)

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, item):
        fname1, fname2 = self.train_files[item]

        img, mask = cv2.imread(fname1), cv2.imread(fname2)
        img, mask = cv2.resize(img, self.input_size), cv2.resize(mask, self.input_size)

        if self.choice == 'train':
            if random.random() < 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
        img = self.transform(img)
        mask = self.tensor(mask[:, :, :1])
        return img, mask, fname1, fname2


def train(args):
    gpu_id = args.gpu_id
    epochs = 60
    batch_size = 4
    lr = 1e-4
    print('epochs:', epochs)
    print('batch_size:', batch_size)
    print('lr:', lr)
    print('gpu_id:', gpu_id)

    # init dataset
    train_dataset = IID_Dataset(choice='train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = IID_Dataset(choice='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print('train_dataset_len:', len(train_dataset))
    print('train_loader_len:', len(train_loader))
    print('test_dataset_len:', len(test_dataset))
    print('test_loader_len:', len(test_loader))

    # init model
    device = torch.device('cuda:' + str(gpu_id))
    model = VideoInpaintingDetectionNet(bilinear=True, sobel=True, dct=True, bayarconv=True)
    model = model.to(device=device)

    # init loss function
    loss_function = FocalLoss()

    # init optim
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    lr_sche = StepLR(optimizer, step_size=10, gamma=0.1)

    # init val_info
    val_info = []

    max_iou = 0.0
    max_iou_index = 0
    max_f1 = 0.0
    max_f1_index = 0

    # training
    for epoch in range(epochs):
        # train
        model.train()
        epoch_train_loss = 0.0
        epoch_train_loss1 = 0.0
        for idx, (image, label, image_path, label_path) in enumerate(train_loader):
            image_torch, label_torch = image.float().to(device), label.float().to(device)
            predict = model(image_torch)
            optimizer.zero_grad()
            loss = loss_function(predict, label_torch)
            loss.backward()
            optimizer.step()
            iter_loss = loss.cpu().float()
            epoch_train_loss = epoch_train_loss + iter_loss
            epoch_train_loss1 = epoch_train_loss + iter_loss
            time = datetime.datetime.now().strftime('%H:%M:%S')
            if (idx + 1) % 50 == 0 and idx != 0:
                epoch_train_loss = epoch_train_loss / 50
                log_string = '[%s] epoch:[%d] iter:[%d]/[%d] loss:[%.5f]' % (time, epoch + 1, idx + 1, len(train_loader), epoch_train_loss)
                print(log_string)
                epoch_train_loss = 0.0
        lr_sche.step()
        epoch_train_loss1 = epoch_train_loss1 / len(train_loader)
        log_string = 'epoch:[%d] train_loss:[%.5f]' % (epoch + 1, epoch_train_loss1)
        print(log_string)
        # test
        model.eval()
        val_loss, val_iou, val_f1 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for idx, (image, label, image_path, label_path) in enumerate(test_loader):
                image_torch, label_torch = image.float().to(device), label.float().to(device)
                predict = model(image_torch)
                loss = loss_function(predict, label_torch)
                iter_loss = loss.cpu().float()
                val_loss = val_loss + iter_loss
                predict_mask = predict[0, 0, ...].cpu().detach().numpy()
                predict_mask = np.where(predict_mask <= 0.5, 0, 1).flatten()
                label = label.cpu().detach().numpy().astype('int').flatten()
                iou, f1 = IOU(predict_mask, label), F1(predict_mask, label)
                val_iou = val_iou + iou
                val_f1 = val_f1 + f1
        val_loss = val_loss / len(test_loader)
        val_iou = val_iou / len(test_loader)
        val_f1 = val_f1 / len(test_loader)
        torch.save(model, os.path.join('./ckpt/1/', str(epoch + 1) + '.pth'))
        log_string = 'epoch:[%d] val_loss:[%.5f] val_iou:[%.5f] val_f1:[%.5f]' % (epoch + 1, val_loss, val_iou, val_f1)

        print(log_string)
        val_info.append(log_string)
        if val_iou > max_iou:
            max_iou = val_iou
            max_iou_index = epoch + 1
        if val_f1 > max_f1:
            max_f1 = val_f1
            max_f1_index = epoch + 1

    for i in val_info:
        print(i)
    print('max_iou:[%.5f] max_iou_index:[%d]' % (max_iou, max_iou_index))
    print('max_f1:[%.5f] max_f1_index:[%d]' % (max_f1, max_f1_index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VideoInpaintingDetection')
    parser.add_argument('--gpu_id', type=int, required=True, help='gpu id')
    args = parser.parse_args()
    train(args)

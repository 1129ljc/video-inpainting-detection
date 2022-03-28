import os
import cv2
import torch
import random
import argparse
import datetime
import numpy as np
from model import HighPassResNetV2, FocalLoss
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils import F1, IOU
from torchvision import transforms

class IID_Dataset(Dataset):
    def __init__(self, choice='test'):
        self.input_size = (512, 512)
        self.image_path = '/home/dell/soft/ljc_methods/video_inpainting_detection/dataset/dstt_train/'
        self.mask_path = '/home/dell/soft/ljc_methods/InpaintingForensics/video_dataset/mask_white/'
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
            transforms.ToTensor()
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



def test():
    save = '/home/dell/soft/ljc_methods/video_inpainting_detection/dataset/dstt_train_result/'
    ckpt = '/home/dell/soft/ljc_methods/video_inpainting_detection/hp_fcn_own/save/dstt/d2/19.pt'
    gpu_id = 0
    filter_type = 'd2'
    if not os.path.exists(save):
        os.makedirs(save)

    print('save:', save)
    print('ckpt:', ckpt)
    print('gpu_id:', gpu_id)
    print('filter_type:', filter_type)

    # init dataset
    test_dataset = IID_Dataset(choice='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print('test_dataset_len:', len(test_dataset))
    print('test_loader_len:', len(test_loader))

    # init model
    device = torch.device('cuda:' + str(gpu_id))
    model = HighPassResNetV2(filter_type)
    model.load_state_dict(torch.load(ckpt))
    model = model.to(device=device)

    # testing
    model.eval()
    val_iou, val_f1 = 0.0, 0.0
    with torch.no_grad():
        for idx, (image, label, image_path, label_path) in enumerate(test_loader):
            image_torch, label_torch = image.float().to(device), label.float().to(device)
            name = str(image_path).split('/')[-2]
            file_name = str(image_path).split('/')[-1][:-3]
            predict = model(image_torch)
            predict_mask = predict[0, 0, ...].cpu().detach().numpy()
            predict_mask_image = predict_mask * 255.
            if not os.path.exists(os.path.join(save, name)):
                os.mkdir(os.path.join(save, name))
            cv2.imwrite(os.path.join(save, name, file_name), predict_mask_image)
            print(os.path.join(save, name, file_name))
            predict_mask = np.where(predict_mask <= 0.5, 0, 1).flatten()
            label = label.cpu().detach().numpy().astype('int').flatten()
            iou, f1 = IOU(predict_mask, label), F1(predict_mask, label)
            val_iou = val_iou + iou
            val_f1 = val_f1 + f1
    val_iou = val_iou / len(test_loader)
    val_f1 = val_f1 / len(test_loader)
    log_string = 'epoch:[19] val_iou:[%.5f] val_f1:[%.5f]' % (val_iou, val_f1)
    print(log_string)


if __name__ == '__main__':
    test()
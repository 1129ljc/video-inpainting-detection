import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from postprocessing import post_deal

class IID_Dataset(Dataset):
    def __init__(self, dataset):
        self.input_size = (512, 512)
        self.image_path = dataset
        self.train_files = []
        names = os.listdir(self.image_path)
        for i in range(len(names)):
            name = names[i]
            image_name_dir = os.path.join(self.image_path, name)
            image_files = sorted(os.listdir(image_name_dir))
            image_num = len(image_files)
            for j in range(image_num):
                image_file = os.path.join(image_name_dir, image_files[j])
                self.train_files.append(image_file)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, item):
        fname1 = self.train_files[item]

        img = cv2.imread(fname1)
        img = cv2.resize(img, self.input_size)
        img = self.transform(img)
        return img, fname1


def test(args):
    dataset = args['dataset']
    ckpt = args['ckpt']
    gpu_id = args['gpu_id']
    save = args['save']

    device = torch.device('cuda:' + str(gpu_id))
    model = torch.load(ckpt)
    model = model.to(device=device)
    model.eval()

    test_dataset = IID_Dataset(dataset)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    num_label = []
    with torch.no_grad():
        for idx, (image, image_path) in enumerate(test_loader):
            image_torch = image.float().to(device)
            predict = model(image_torch)
            predict_mask = predict[0, 0, ...].cpu().detach().numpy()
            # predict_mask_image = predict_mask * 255.
            predict_mask_image = np.zeros([512, 512, 3])
            predict_mask_image[..., 0] = predict_mask * 255.
            predict_mask_image[..., 1] = predict_mask * 255.
            predict_mask_image[..., 2] = predict_mask * 255.
            num_labels, output = post_deal(predict_mask_image)
            save0 = str(image_path)[:-3].split('/')
            save1, filename = save0[-2], save0[-1]
            if not os.path.exists(os.path.join(save, save1)):
                os.mkdir(os.path.join(save, save1))
            cv2.imwrite(os.path.join(save, save1, filename), output)
            print(os.path.join(save, save1, filename))
            num_label.append(num_labels)
    num_label_result = sum(num_label)/len(num_label)
    return num_label_result


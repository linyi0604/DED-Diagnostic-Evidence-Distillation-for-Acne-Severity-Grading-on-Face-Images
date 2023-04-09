import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np

class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        # reading img file from file
        fp = open(img_filename, 'r')

        self.img_filename = []
        self.labels = []
        self.lesion_numers = []
        for line in fp.readlines():
            filename, label, lesion_numer = line.split()
            self.img_filename.append(filename)
            self.labels.append(int(label))
            self.lesion_numers.append(int(lesion_numer))
        fp.close()


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index])).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        name = self.img_filename[index]
        label = self.labels[index]
        lesion_numer = self.lesion_numers[index]
        return img, label, lesion_numer

    def __len__(self):
        return len(self.img_filename)


class DatasetProcessing_image_mask(Dataset):
    def __init__(self, data_path, mask_path, img_filename, transform=None):
        self.img_path = data_path
        self.mask_path = mask_path
        self.transform = transform
        # reading img file from file
        fp = open(img_filename, 'r')

        self.img_filename = []
        self.labels = []
        self.lesion_numers = []
        for line in fp.readlines():
            filename, label, lesion_numer = line.split()
            self.img_filename.append(filename)
            self.labels.append(int(label))
            self.lesion_numers.append(int(lesion_numer))
        fp.close()


    def __getitem__(self, index):
        img = Image.open(self.img_path + self.img_filename[index]).convert("RGB")
        mask = Image.open(self.mask_path + self.img_filename[index]).convert("1")
        # img = np.transpose(np.array(img), (2, 0, 1))
        # mask = np.transpose(np.array(mask), (2, 0, 1))
        # img = torch.from_numpy(img)
        # mask = torch.from_numpy(mask)
        # print(mask.shape)
        # print(img.shape)
        # raise

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
        name = self.img_filename[index]
        severity = self.labels[index]
        lesion_numer = self.lesion_numers[index]
        x = torch.cat([img, mask], dim=0)
        return x, severity, lesion_numer

    def __len__(self):
        return len(self.img_filename)


class DatasetProcessing_teacher_student(Dataset):
    def __init__(self, data_path, mask_path, img_filename, transform=None):
        self.img_path = data_path
        self.mask_path = mask_path
        self.transform = transform
        # reading img file from file
        fp = open(img_filename, 'r')

        self.img_filename = []
        self.labels = []
        self.lesion_numers = []
        for line in fp.readlines():
            filename, label, lesion_numer = line.split()
            self.img_filename.append(filename)
            self.labels.append(int(label))
            self.lesion_numers.append(int(lesion_numer))
        fp.close()


    def __getitem__(self, index):
        img = Image.open(self.img_path + self.img_filename[index]).convert("RGB")
        mask = Image.open(self.mask_path + self.img_filename[index]).convert("1")
        # img = np.transpose(np.array(img), (2, 0, 1))
        # mask = np.transpose(np.array(mask), (2, 0, 1))
        # img = torch.from_numpy(img)
        # mask = torch.from_numpy(mask)
        # print(mask.shape)
        # print(img.shape)
        # raise

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
        name = self.img_filename[index]
        severity = self.labels[index]
        lesion_numer = self.lesion_numers[index]
        x = torch.cat([img, mask], dim=0)
        return x, img, severity, lesion_numer

    def __len__(self):
        return len(self.img_filename)



class DatasetProcessing_teacher_student_augmentation(Dataset):
    def __init__(self, data_path, mask_path, img_filename, transform=None, transform_image=None):
        self.img_path = data_path
        self.mask_path = mask_path
        self.transform = transform
        self.transform_image = transform_image
        # reading img file from file
        fp = open(img_filename, 'r')

        self.img_filename = []
        self.labels = []
        self.lesion_numers = []
        for line in fp.readlines():
            filename, label, lesion_numer = line.split()
            self.img_filename.append(filename)
            self.labels.append(int(label))
            self.lesion_numers.append(int(lesion_numer))
        fp.close()


    def __getitem__(self, index):
        img = Image.open(self.img_path + self.img_filename[index]).convert("RGB")
        mask = Image.open(self.mask_path + self.img_filename[index]).convert("1")

        if self.transform is not None:
            img_teacher = self.transform(img)
            mask = self.transform(mask)
        else:
            raise

        if self.transform_image is not None:
            img_student = self.transform_image(img)
        else:
            img_student = img_teacher

        name = self.img_filename[index]
        severity = self.labels[index]
        lesion_numer = self.lesion_numers[index]
        x = torch.cat([img_teacher, mask], dim=0)
        return x, img_student, severity, lesion_numer

    def __len__(self):
        return len(self.img_filename)

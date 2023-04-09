import os
from torchvision import transforms
from dataset import dataset_processing
from torch.utils.data import DataLoader
from model import vggnet
import torch
from torch import nn
from utils.utils import Logger, AverageMeter, time_to_str
from timeit import default_timer as timer
import numpy as np
from utils.report import report_precision_se_sp_yi


class Config(object):
    cross_validation_index = ["1"]
    # cross_validation_index = ["0", "1", "2", "3", "4"]
    def __init__(self, cross_val_index):
        self.gpu_id = "0"
        self.image_path = "../data/JPEGImages/"
        self.mask_path = "../data/mask/"
        self.train_mapping_path = '../data/NNEW_trainval_' + cross_val_index + '.txt'
        self.test_mapping_path = '../data/NNEW_test_' + cross_val_index + '.txt'
        self.model_save_path = "../results/teacher_%s.pkl"%cross_val_index
        self.log_file = "../results/log_teacher_"
        self.num_workers = 16
        self.network_input_size = (224, 224)
        self.batch_size = 16
        self.class_num = 4
        self.max_counting_num = 80
        self.learning_rate = 0.001
        self.epoch = 5
        # self.epoch = 120
        self.input_channel = 4
        self.lambda_g = 0.5
        self.lambda_c = 0.5



def train_test(cross_val_index):

    config = Config(cross_val_index)
    log = Logger()
    log.open(config.log_file + "%s.log"%cross_val_index, mode="a")
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    dset_train = dataset_processing.DatasetProcessing_image_mask(
        config.image_path, config.mask_path, config.train_mapping_path,
        transform=transforms.Compose([
                transforms.Resize(config.network_input_size),
                transforms.ToTensor()
            ]))

    dset_test = dataset_processing.DatasetProcessing_image_mask(
        config.image_path, config.mask_path, config.test_mapping_path,
        transform=transforms.Compose([
                transforms.Resize(config.network_input_size),
                transforms.ToTensor(),
            ]))

    train_loader = DataLoader(dset_train,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)

    test_loader = DataLoader(dset_test,
                             batch_size=config.batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=config.num_workers)

    model = vggnet.VGG16_image_mask_to_level_couting_regression(
        input_channel=config.input_channel,
        class_num=config.class_num,
        max_distribution_num=config.max_counting_num,
        pretrained=True).cuda()
    # model.load_state_dict(torch.load(config.model_save_path))
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config.learning_rate, weight_decay=5e-4, momentum=0.9)
    loss_func_grading = nn.CrossEntropyLoss()
    loss_func_counting = nn.CrossEntropyLoss()

    start = timer()
    max_acc = 0
    best_report = ""

    for epoch in range(config.epoch):
        losses = AverageMeter()
        for step, (img, label, lesion_numer) in enumerate(train_loader):

            lesion_numer = lesion_numer.cuda()
            img = img.cuda()
            label = label.cuda()
            # train
            model.train()
            pre_classification, pre_regression = model(img)

            loss_grading = loss_func_grading(pre_classification, label)
            loss_counting = loss_func_counting(pre_regression, lesion_numer)
            loss = config.lambda_g*loss_grading + config.lambda_c*loss_counting
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            losses.update(loss.item(), img.size(0))
        message = '%s  | %0.3f | %0.3f | %s\n' % (
            "train", epoch,
            losses.avg,
            time_to_str((timer() - start), 'min'))

        log.write(message)

        # test process
        with torch.no_grad():
            test_loss = 0
            test_corrects = 0
            y_true = np.array([])
            y_pred = np.array([])
            AE = 0
            model.eval()
            for step, (test_img, test_label, test_lesions_number) in enumerate(test_loader):
                test_lesions_number = test_lesions_number.cuda()
                test_img = test_img.cuda()
                test_label = test_label.cuda()
                y_true = np.hstack((y_true, test_label.data.cpu().numpy()))

                b_pre_cls, b_pre_re = model(test_img)

                loss = loss_func_grading(b_pre_cls, test_label)
                loss_counting = loss_func_counting(b_pre_re, test_lesions_number)
                test_loss += loss.data + loss_counting.item()

                _, preds = torch.max(b_pre_cls, 1)
                _, re_pre = torch.max(b_pre_re, 1)
                AE += np.sum(np.abs((re_pre - test_lesions_number).cpu().numpy()))


                y_pred = np.hstack((y_pred, preds.data.cpu().numpy()))

                batch_corrects = torch.sum((preds == test_label)).data.cpu().numpy()
                test_corrects += batch_corrects

            # test_loss = test_loss.float() / len(test_loader)
            test_acc = test_corrects / len(test_loader.dataset)  # 3292  #len(test_loader)
            MAE = AE / len(dset_test)
            _, _, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred, y_true)

            pre_se_sp_yi_report = str(pre_se_sp_yi_report) + "\n MAE: %.4f" % MAE

            if test_acc > max_acc:
                max_acc = test_acc
                best_report = pre_se_sp_yi_report + "\n"
                torch.save(model.state_dict(), config.model_save_path)
            if True:
                log.write(pre_se_sp_yi_report + '\n')
                log.write("best result until now: \n")
                log.write(best_report + '\n')
        log.write("best result: \n")
        log.write(best_report + '\n')
        print(epoch)
        print(pre_se_sp_yi_report)
        print(best_report)



if __name__ == '__main__':
    for cross_val_index in Config.cross_validation_index:
        train_test(cross_val_index)

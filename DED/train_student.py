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
from torch.nn.functional import softmax


class Config(object):
    cross_validation_index = ["1"]
    # cross_validation_index = ["0", "1", "2", "3", "4"]
    def __init__(self, cross_val_index):
        self.gpu_id = "0"
        self.image_path = "../data/JPEGImages/"
        self.mask_path = "../data/mask/"
        self.train_mapping_path = '../data/NNEW_trainval_' + cross_val_index + '.txt'
        self.test_mapping_path = '../data/NNEW_test_' + cross_val_index + '.txt'
        self.model_save_path = "../results//student_%s"%(cross_val_index)+".pkl"
        self.teacher_model_load_path = "../results//teacher_"+cross_val_index+".pkl"
        self.log_file = "../results/log_student_%s"%(cross_val_index)+".log"
        self.num_workers = 16
        self.resize = (240, 240)
        self.network_input_size = (224, 224)
        self.batch_size = 16
        self.class_num = 4
        self.max_counting_num = 80
        self.learning_rate = 0.001
        self.epoch = 5
        # self.epoch = 120
        self.input_channel = 4
        self.lambda_g = 0.7
        self.lambda_d = 0.3



def train_test(cross_val_index):

    config = Config(cross_val_index)
    log = Logger()
    log.open(config.log_file, mode="a")
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    dset_train = dataset_processing.DatasetProcessing_teacher_student_augmentation(
        config.image_path, config.mask_path, config.train_mapping_path,
        transform=transforms.Compose([
            transforms.Resize(config.network_input_size),
            transforms.ToTensor(),
        ]),
        transform_image=transforms.Compose([
                transforms.RandomRotation(20),
                transforms.Resize(config.resize),
                transforms.RandomCrop(config.network_input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
        ]))

    dset_test = dataset_processing.DatasetProcessing_teacher_student_augmentation(
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

    teacher = vggnet.Teacher(
        input_channel=config.input_channel,
        class_num=config.class_num,
        max_distribution_num=config.max_counting_num,
        pretrained=True).cuda()
    teacher.load_state_dict(torch.load(config.teacher_model_load_path))

    model = vggnet.Student(
        config.class_num,
        pretrained=True).cuda()
    # model.load_state_dict(torch.load(config.model_save_path))

    optimizer = torch.optim.SGD(params=model.parameters(), lr=config.learning_rate, weight_decay=5e-4, momentum=0.9)
    loss_fun_grading = nn.CrossEntropyLoss()
    loss_func_distillation = nn.L1Loss()

    start = timer()
    max_acc = 0
    best_report = ""

    teacher.eval()
    for epoch in range(config.epoch):

        losses = AverageMeter()
        losses_cls = AverageMeter()
        losses_distribution = AverageMeter()
        model.train()
        for step, (teacher_signal, img, label, lesion_numer) in enumerate(train_loader):
            lesion_numer = lesion_numer.cuda()
            teacher_signal = teacher_signal.cuda()
            img = img.cuda()
            label = label.cuda()
            # train
            teacher_distribution, teacher_cls, teacher_reg = teacher(teacher_signal)

            student_distribution, pre_classification = model(img)

            loss_distillation = loss_func_distillation(student_distribution, teacher_distribution.detach())
            loss_grading = loss_fun_grading(pre_classification, label)

            loss = config.lambda_g*loss_grading + config.lambda_d*loss_distillation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            losses_distribution.update(loss_distillation.item(), img.size(0))
            losses_cls.update(loss_grading.item(), img.size(0))
            losses.update(loss.item(), img.size(0))



        message = '%s epoch: %s  ' \
                  '| loss: %0.3f ' \
                  '| cls_loss: %0.3f ' \
                  '| disttilation_loss: %0.3f  ' \
                  '| %s\n' % (
            "train", epoch,
            losses.avg,
            losses_cls.avg,
            losses_distribution.avg,
            time_to_str((timer() - start), 'min'))

        log.write(message)

        # test process
        with torch.no_grad():
            losses_distribution = AverageMeter()
            losses_cls = AverageMeter()
            test_corrects = 0
            y_true = np.array([])
            y_pred = np.array([])
            model.eval()
            for step, (test_teacher_signal, test_img, test_label, test_lesions_number) in enumerate(test_loader):

                test_teacher_signal = test_teacher_signal.cuda()
                test_img = test_img.cuda()
                test_label = test_label.cuda()
                y_true = np.hstack((y_true, test_label.data.cpu().numpy()))
                teacher_distribution, teacher_classification, _ = teacher(test_teacher_signal)
                student_distribution, b_pre_cls = model(test_img)

                loss = loss_fun_grading(b_pre_cls, test_label)
                loss_re = loss_func_distillation(student_distribution, teacher_distribution)
                losses_distribution.update(loss_re.item(), test_img.size(0))
                losses_cls.update(loss.item(), test_img.size(0))

                _, preds = torch.max(b_pre_cls, 1)



                y_pred = np.hstack((y_pred, preds.data.cpu().numpy()))

                batch_corrects = torch.sum((preds == test_label)).data.cpu().numpy()
                test_corrects += batch_corrects


            test_acc = test_corrects / len(test_loader.dataset)  # 3292  #len(test_loader)
            _, _, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred, y_true)


            pre_se_sp_yi_report = str(pre_se_sp_yi_report) + \
                                  "\n cls_loss: %.4f, distillation_loss:%.4f" \
                                  % (losses_cls.avg, losses_distribution.avg)

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




if __name__ == '__main__':

    for cross_val_index in Config.cross_validation_index:
        train_test(cross_val_index)

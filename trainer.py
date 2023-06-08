import matplotlib.pyplot as plt
import PIL
from utils import test_single_volume
from torchvision import transforms
from utils import DiceLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.modules.loss import BCELoss
from torch.nn.modules.loss import CrossEntropyLoss
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import time
import sys
import random
import logging
import argparse
import os
import cv2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


def trainer_vfss(args, model, snapshot_path):
    from datasets.dataset_vfss import VFSS_dataset, RandomGenerator, RandomGenerator_test
    logging.basicConfig(filename=snapshot_path + "/logtrain_" + args.test_case + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    n_batch = args.batch_size/args.n_gpu

    db_train = VFSS_dataset(base_dir=args.root_path, split="train",
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    db_valid = VFSS_dataset(base_dir=args.root_path, split="valid",
                            transform=transforms.Compose(
                                [RandomGenerator_test(output_size=[args.img_size, args.img_size])]))

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of valid set is: {}".format(len(db_valid)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    validloader = DataLoader(db_valid, batch_size=batch_size, shuffle=False)

    # if args.n_gpu > 1:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = nn.DataParallel(model).to(device)

    model.train()

    ce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/tensorboard_' + args.test_case)
    iter_num = 0
    max_epoch = args.max_epochs
    # max_epoch = max_iterations // len(trainloader) + 1
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(
        len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    best_loss = 100000
    patience_limit = 20  # 20  Validation loss ½ Early Stop
    patience_check = 0
    total_loss = 0

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):  # 496
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs, outputs2 = model(image_batch) 
            
            Bolus1 = dice_loss(outputs[:, 0, :, :], label_batch[:, 0, :, :])
            Cervical1 = dice_loss(outputs[:, 1, :, :], label_batch[:, 1, :, :])
            Hyoid1 = dice_loss(outputs[:, 2, :, :], label_batch[:, 2, :, :])
            Mandible1 = dice_loss(outputs[:, 3, :, :], label_batch[:, 3, :, :])
            Soft1 = dice_loss(outputs[:, 4, :, :], label_batch[:, 4, :, :])
            Vocal1 = dice_loss(outputs[:, 5, :, :], label_batch[:, 5, :, :])

            Bolus2 = dice_loss(outputs2[:, 0, :, :], label_batch[:, 0, :, :])
            Cervical2 = dice_loss(outputs2[:, 1, :, :], label_batch[:, 1, :, :])
            Hyoid2 = dice_loss(outputs2[:, 2, :, :], label_batch[:, 2, :, :])
            Mandible2 = dice_loss(outputs2[:, 3, :, :], label_batch[:, 3, :, :])
            Soft2 = dice_loss(outputs2[:, 4, :, :], label_batch[:, 4, :, :])
            Vocal2 = dice_loss(outputs2[:, 5, :, :], label_batch[:, 5, :, :])

            loss_dice1 = Bolus1*0.25 + Cervical1*2.5 + Hyoid1 * \
                0.25 + Mandible1*2.5 + Soft1*0.25 + Vocal1*0.25
            loss_dice2 = Bolus2*2.5 + Cervical2*0.7 + Hyoid2 * \
                0.7 + Mandible2*0.7 + Soft2*0.7 + Vocal2*0.7
            loss = loss_dice1 + loss_dice2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/Bolus2_loss', Bolus2, iter_num)
            writer.add_scalar('info/Cervical2_loss', Cervical2, iter_num)
            writer.add_scalar('info/Hyoid2_loss', Hyoid2, iter_num)
            writer.add_scalar('info/Mandible2_loss', Mandible2, iter_num)
            writer.add_scalar('info/Soft2_loss', Soft2, iter_num)
            writer.add_scalar('info/Vocal2_loss', Vocal2, iter_num)
            logging.info('iteration %d : total loss : %f : loss_dice1 : %f : loss_dice2 : %f' % (
                iter_num, loss, loss_dice1.item(), loss_dice2.item()))

            if iter_num % 45 == 0 and iter_num != 0:
                model.eval()
                total_loss = 0
                with torch.no_grad():
                    for _, sampled_batch in enumerate(validloader):
                        image_batch, label_batch, case_batch = sampled_batch[
                            'image'], sampled_batch['label'], sampled_batch['case_name']
                        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

                        outputs, outputs2 = model(image_batch)

                        # basic4(cs,m 6-> b 6) ê¸°ì??
                        Bolus1 = dice_loss(
                            outputs[:, 0, :, :], label_batch[:, 0, :, :])
                        Cervical1 = dice_loss(
                            outputs[:, 1, :, :], label_batch[:, 1, :, :])
                        Hyoid1 = dice_loss(
                            outputs[:, 2, :, :], label_batch[:, 2, :, :])
                        Mandible1 = dice_loss(
                            outputs[:, 3, :, :], label_batch[:, 3, :, :])
                        Soft1 = dice_loss(
                            outputs[:, 4, :, :], label_batch[:, 4, :, :])
                        Vocal1 = dice_loss(
                            outputs[:, 5, :, :], label_batch[:, 5, :, :])

                        Bolus2 = dice_loss(
                            outputs2[:, 0, :, :], label_batch[:, 0, :, :])
                        Cervical2 = dice_loss(
                            outputs2[:, 1, :, :], label_batch[:, 1, :, :])
                        Hyoid2 = dice_loss(
                            outputs2[:, 2, :, :], label_batch[:, 2, :, :])
                        Mandible2 = dice_loss(
                            outputs2[:, 3, :, :], label_batch[:, 3, :, :])
                        Soft2 = dice_loss(
                            outputs2[:, 4, :, :], label_batch[:, 4, :, :])
                        Vocal2 = dice_loss(
                            outputs2[:, 5, :, :], label_batch[:, 5, :, :])

                        loss_dice1 = Bolus1*0.25 + Cervical1*2.5 + Hyoid1 * \
                            0.25 + Mandible1*2.5 + Soft1*0.25 + Vocal1*0.25
                        # loss_dice2 = Bolus2 + Cervical2*0.05 + Hyoid2*0.05 + Mandible2*0.05 + Soft2*0.05 + Vocal2*0.4 # basic5
                        # loss_dice1 = Bolus1 + Cervical1 + Hyoid1 + Mandible1 + Soft1 + Vocal1
                        loss_dice2 = Bolus2*2.5 + Cervical2*0.7 + Hyoid2 * \
                            0.7 + Mandible2*0.7 + Soft2*0.7 + Vocal2*0.7
                        loss = loss_dice2

                        writer.add_scalar(
                            'info/valid_total_loss_dice2', loss, iter_num)
                        writer.add_scalar(
                            'info/valid_Bolus2_loss', Bolus2, iter_num)
                        writer.add_scalar(
                            'info/valid_Cervical2_loss', Cervical2, iter_num)
                        writer.add_scalar(
                            'info/valid_Hyoid2_loss', Hyoid2, iter_num)
                        writer.add_scalar(
                            'info/valid_Mandible2_loss', Mandible2, iter_num)
                        writer.add_scalar(
                            'info/valid_Soft2_loss', Soft2, iter_num)
                        writer.add_scalar(
                            'info/valid_Vocal2_loss', Vocal2, iter_num)
                        logging.info('iteration %d : valid_loss_dice1 : %f : valid_bolus_loss_dice2 : %f' % (
                            iter_num, loss_dice1.item(), Bolus2.item()))

                        total_loss += Bolus2

                    # Early Stopping conditions
                    if total_loss > best_loss:
                        patience_check += 1
                    else:  # Make checkpoint at currently best model
                        save_mode_path = os.path.join(
                            snapshot_path, args.test_case + '_bestmodel_epoch:' + str(epoch_num) + 'iternum:'+str(iter_num)+'.pth')
                        torch.save(model.module.state_dict(),
                                    save_mode_path)
                        logging.info(
                            "SAVE CHECKPOINT: {}".format(save_mode_path))
                        best_loss = total_loss
                        patience_check = 0
                model.train()

        if (epoch_num % 50 == 0 and epoch_num != 0) or epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(
                snapshot_path, args.test_case + '_' + str(epoch_num+1) + '.pth')
            torch.save(model.module.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()

    writer.close()
    return "Training Finished!"


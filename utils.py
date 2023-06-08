
import sys
import cv2
import torchvision.transforms as T
import torch.nn as nn
from medpy import metric
import torch
import numpy as np
from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        assert inputs.size() == target.size(
        ), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        sigmoid = nn.Sigmoid()
        inputs = sigmoid(inputs)
        loss = self._dice_loss(inputs, target)

        return loss


def calculate_metric_percase(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, grad_save_path=None, case=None, z_spacing=1):
    sigmoid = nn.Sigmoid()
    mse = nn.MSELoss()
    image, label = image.cuda(), label.cuda()
    net.eval()
    # print('image: ', image.shape) # (1,3,224,224)
    with torch.no_grad():
        first_output, second_output = net(image)

        # m = nn.Threshold(0.4, 0)
        # outputs = m(sigmoid(first_output))
        # outputs2 = m(sigmoid(second_output))

        # outputs = sigmoid(first_output)
        # outputs = (outputs>0.3).float()
        # outputs2 = sigmoid(second_output)
        # outputs2 = (outputs2>0.3).float()

        outputs = torch.round(sigmoid(first_output))
        outputs2 = torch.round(sigmoid(second_output))
        # outputs2 = sigmoid(second_output)
        # outputs2 = (outputs2>0.3).float()
        # outputs = outputs > 0.1

    outputs, outputs2, label = outputs.cpu().detach().numpy(
    ), outputs2.cpu().detach().numpy(), label.cpu().detach().numpy()
    metric_list = []
    metric_list2 = []
    for i in range(classes):
        metric_list.append(calculate_metric_percase(
            outputs[0][i], label[0][i]))
        metric_list2.append(calculate_metric_percase(
            outputs2[0][i], label[0][i]))

    if test_save_path is not None:
        for i in range(classes):
    #         visualization(outputs[0][i], label[0][i],
    #                       case, test_save_path, i, 1)
    #         visualization(outputs2[0][i], label[0][i],
    #                       case, test_save_path, i, 2)
        # image : 1, 3 ,224 ,224  / label[0][i] : 224,224
            grad_cam_application(net, image, label[0][i], case, grad_save_path, i)
    return metric_list, metric_list2


def visualization(pred, label, path, s_path, n_class, n):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))  # 224 224 3

    # TP = pred
    # print(pred.shape)
    TP = np.zeros(pred.shape)

    for i in range(0, pred.shape[0]):
        for j in range(0, pred.shape[0]):
            if (pred[i, j] == 1) and (label[i, j] != 0):
                TP[i, j] = 1
            elif (label[i, j] == 0) and (pred[i, j] == 1):
                # print("FP")
                TP[i, j] = 2  # FP
                # print(TP[i, j])
            elif (label[i, j] != 0) and (pred[i, j] == 0):
                # print("FN")
                TP[i, j] = 3  # FN
                # print(TP[i, j])

    for i in range(0, pred.shape[0]):
        for j in range(0, pred.shape[0]):
            if TP[i, j] == 1:  # TP
                image[i, j, 0] = 255
                image[i, j, 1] = 0
                image[i, j, 2] = 0
            elif TP[i, j] == 2:  # FP
                # print("--FP")
                image[i, j, 0] = 0
                image[i, j, 1] = 255
                image[i, j, 2] = 0
            elif TP[i, j] == 3:  # FN
                # print("--FN")
                image[i, j, 0] = 0
                image[i, j, 1] = 0
                image[i, j, 2] = 255

    name = ['bolus', 'cervical spine', 'hyoid bone',
            'mandible', 'soft tissue', 'vocal folds']
    path = path.split("/")
    path = path[len(path)-1]
    if(n == 1):
        cv2.imwrite(s_path + '/' + path + "_" +
                    name[n_class] + 'label.png', image)
    if(n == 2):
        cv2.imwrite(s_path + '/' + path + "_" +
                    name[n_class] + 'logit2' + 'label.png', image)


def grad_cam_application(net, image, label, path, g_path, n_class):

    #target_layers = [net.transformer.embeddings.hybrid_model.root.relu]
    #target_layers = [net.transformer.embeddings.hybrid_model.body.block1.unit3.relu]
    # target_layers = [net.transformer.embeddings.hybrid_model.body.block1.unit3.conv3]
    #target_layers = [net.transformer.embeddings.hybrid_model.body.block2.unit4.relu]
    target_layers = [net.transformer2.embeddings.hybrid_model.body.block2.unit4.relu]
    # target_layers = [
    #     net.transformer.embeddings.hybrid_model.body.block3.unit9.relu]

    # target_layers = [net.transformer.encoder.layer[5].attn.query]
    #target_layers = [net.transformer.encoder.layer[11].attn.softmax]

    #target_layers = [net.decoder.conv_more[0]]
    # target_layers = [net.decoder.blocks[0].conv2[2]]
    #target_layers = [net.decoder.blocks[1].conv2[2]]
    #target_layers = [net.decoder.blocks[2].conv2[2]]
    #target_layers = [net.decoder.blocks[3].conv2[2]]

    print('target_layers: ', target_layers)
    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
    targets = [SemanticSegmentationTarget(n_class, label)]

    # print("image4: ", image.shape)  # (1, 3, 224, 224)
    # print("label", label.shape) # (6, 224, 224)
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=image, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    print('grayscale_cam: ', grayscale_cam.shape)

    grayscale_cam = grayscale_cam[0, :]
    print(grayscale_cam.shape)
    print('after grayscale_cam: ', grayscale_cam.shape)
    image = image.permute(0, 2, 3, 1)
    image = image.cpu().numpy()
    image[image > 1] = 1
    # print('^^:', image[0].shape)
    # visualization = show_cam_on_image(image[0], grayscale_cam, use_rgb=True)
    # print("kkkkkk", visualization.shape)
    name = ['bolus', 'cervical spine', 'hyoid bone',
            'mandible', 'soft tissue', 'vocal folds']

    path = path.split("/")
    path = path[len(path)-1].split(".")[0]
    print(g_path + '/' + path + "_" + name[n_class] + '.png')
    cv2.imwrite(g_path + '/' + path + "_" +
                name[n_class] + '.png', visualization)

import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from math import ceil
import datetime


def rgb2mask(img, color2index):
    W = np.power(256, [[0], [1], [2]])
    img_id = img.dot(W).squeeze(-1)
    values = np.unique(img_id)

    mask = np.zeros(img_id.shape)

    for i, c in enumerate(values):
        try:
            mask[img_id == c] = color2index[tuple(img[img_id == c][0])]
        except:
            pass

    return mask


def my_trainer(model, cfg, ikDataset, stop, step_fct, writer, seed=10):
    batch_size = cfg.batch_size
    img_size = cfg.img_size
    num_classes = cfg.num_classes
    base_lr = cfg.base_lr
    max_iterations = cfg.max_iter
    snapshot_path = cfg.output_path
    split_train_test = cfg.split_train_test
    eval_period = cfg.eval_period
    n_gpu = 1
    batch_size = batch_size * n_gpu

    # empirical values for warmup
    warmup_iters = max_iterations // 3
    warmup_factor = 0.001
    transformations = [Nphwc2tensorchw(), Resize(img_size, img_size)]
    if cfg.pretrain is not None:
        mean = np.array([123.675, 116.280, 103.530], dtype=np.float)
        std = np.array([58.395, 57.120, 57.375], dtype=np.float)
        norm = Normalize(mean=mean, std=std)
        # norm = NormalizeFromPaper()
        unorm = UnNormalize(mean=mean, std=std)
        # preprocess input for resnet pretrained on imagenet
        transformations.append(norm)

    # transformations.append(RandomGenerator(output_size=[img_size, img_size]))

    idx_split = int(len(ikDataset["images"]) * split_train_test)

    random.seed(seed)
    random.shuffle(ikDataset["images"])

    db_train = My_dataset({"metadata": ikDataset["metadata"], "images": ikDataset["images"][:idx_split]},
                          transform=transforms.Compose(transformations))
    db_test = My_dataset({"metadata": ikDataset["metadata"], "images": ikDataset["images"][idx_split:]},
                         transform=transforms.Compose(transformations))

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of test set is: {}".format(len(db_test)))

    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=1, num_workers=0, pin_memory=True,
                            worker_init_fn=worker_init_fn)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    # ce_loss = CrossEntropyLoss()
    # dice_loss = DiceLoss(num_classes)
    hard_pixel_loss = DeepLabCE(top_k_percent_pixels=0.2)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    iter_num = 0
    print("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    max_epoch = ceil(max_iterations / batch_size / len(db_train))
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        if stop():
            break
        for i_batch, sampled_batch in enumerate(trainloader):
            if stop() or iter_num > max_iterations - 1:
                break
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            # loss_ce = ce_loss(outputs, label_batch[:].long())
            # loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss = 0.5 * loss_ce + 0.5 * loss_dice
            loss = hard_pixel_loss(outputs, label_batch[:].long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = get_lr(base_lr, iter_num, max_iterations, warmup_iters, warmup_factor)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            step_fct()
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            print('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 20 == 0:
                image = image_batch[0, 0, :, :, :]
                if cfg.pretrain is not None:
                    image = unorm(image.float())

                writer.add_image('train/Image', image / 255, iter_num, dataformats='CHW')
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0)
                writer.add_image('train/GroundTruth', labs * 50, iter_num)

            if iter_num % eval_period == 0 and len(testloader) > 0:
                _N = num_classes
                conf_matrix = np.zeros((_N, _N), dtype=np.int64)

                for i_batch, sampled_batch in enumerate(testloader):
                    image, gt = sampled_batch['image'], sampled_batch['label']
                    image = image.cuda()
                    gt = gt.cuda()
                    with torch.no_grad():
                        output = model(image)
                        pred = torch.argmax(torch.softmax(output, dim=1), dim=1, keepdim=True)
                    conf_matrix += np.bincount(_N * pred.cpu().reshape(-1) + gt.cpu().reshape(-1),
                                               minlength=_N ** 2).reshape(_N, _N)
                acc = np.full(_N, np.nan, dtype=np.float)
                iou = np.full(_N, np.nan, dtype=np.float)
                tp = conf_matrix.diagonal().astype(np.float)
                pos_gt = np.sum(conf_matrix, axis=0).astype(np.float)
                pos_pred = np.sum(conf_matrix, axis=1).astype(np.float)
                acc_valid = pos_gt > 0

                acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
                iou_valid = (pos_gt + pos_pred) > 0
                union = pos_gt + pos_pred - tp
                iou[acc_valid] = tp[acc_valid] / union[acc_valid]
                macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
                miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
                writer.add_scalar('info/macc', macc, iter_num)
                writer.add_scalar('info/miou', miou, iter_num)

                for class_iou, class_name in zip(iou, ikDataset["metadata"]["category_names"].values()):
                    writer.add_scalar('info/iou-' + class_name, class_iou, iter_num)

    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    iterator.close()

    writer.close()
    return "Training Finished!"


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # image = transforms.functional.normalize(image, mean=self.mean, std=self.std)
        # image = (image-image.min())/(image.max()-image.min())
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        sample = {'image': image, 'label': label}
        return sample


class NormalizeFromPaper(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = (image - image.min()) / (image.max() - image.min())

        sample = {'image': image, 'label': label}
        return sample


def get_lr(base_lr, iter, max_iter, warmup_iters=None, warmup_factor=None):
    if iter >= warmup_iters or warmup_iters is None:
        factor = 1
    else:
        alpha = iter / warmup_iters
        factor = warmup_factor * (1 - alpha) + alpha
    return base_lr * factor * (1.0 - iter / max_iter) ** 0.9


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be un-normalized.
        Returns:
            Tensor: Un-normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub(m).div(s)
        return tensor


class Resize(object):
    def __init__(self, height, width):
        self.h = height
        self.w = width

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = transforms.functional.resize(image, size=(self.h, self.w))
        label = transforms.functional.resize(label.unsqueeze(0), size=(self.h, self.w)).squeeze(0)
        sample = {'image': image, 'label': label}
        return sample


class Nphwc2tensorchw(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = image.permute(0, 3, 1, 2)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class My_dataset(Dataset):
    def __init__(self, ikDataset, transform=None):
        self.transform = transform  # using transform in torch!
        self.sample_list = ikDataset
        if "category_colors" in ikDataset["metadata"]:
            self.labelmap = {k[::-1]: v for v, k in enumerate(ikDataset["metadata"]["category_colors"])}

    def __len__(self):
        return len(self.sample_list["images"])

    def __getitem__(self, idx):

        record = self.sample_list["images"][idx]
        if "segmentation_masks_np" in record:
            image, label = cv2.imread(record["filename"]), record["segmentation_masks_np"]
        else:
            if "semantic_seg_masks_file" in record:
                image, label = cv2.imread(record["filename"]), cv2.imread(record["semantic_seg_masks_file"])
            else:
                if "instance_seg_masks_file" in record:
                    image, label = cv2.imread(record["filename"]), cv2.imread(record["instance_seg_masks_file"])

        if "category_colors" in self.sample_list["metadata"]:
            label = rgb2mask(label, self.labelmap)
        else:
            label = label[:, :, 0]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class DeepLabCE(nn.Module):
    """
    Hard pixel mining with cross entropy loss, for semantic segmentation.
    This is used in TensorFlow DeepLab frameworks.
    Paper: DeeperLab: Single-Shot Image Parser
    Reference: https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/deeplab/utils/train_utils.py#L33  # noqa
    Arguments:
        ignore_label: Integer, label to ignore.
        top_k_percent_pixels: Float, the value lies in [0.0, 1.0]. When its
            value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining.
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, ignore_label=-1, top_k_percent_pixels=1.0, weight=None):
        super(DeepLabCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )

    def forward(self, logits, labels, weights=None):
        if weights is None:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        else:
            # Apply per-pixel loss weights.
            pixel_losses = self.criterion(logits, labels) * weights
            pixel_losses = pixel_losses.contiguous().view(-1)
        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.mean()

        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        return pixel_losses.mean()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
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
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

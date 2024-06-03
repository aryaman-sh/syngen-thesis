import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.transforms import (
    Activationsd,
    AsDiscreted)


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, dummy_target):
        print(x.shape)
        entropy = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        entropy = - entropy.sum()
        return entropy


def train_NormSeg(model1, model2, dataloader, optimizer, loss_func, device):
    """ TODO """
    model1.train()
    model2.train()
    total_loss = 0
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()

        model1.to(device)
        model2.to(device)

        input_img, target = data['image'].to(device), data['label'].to(device)

        norm_input = model1(input_img)
        output = model2(norm_input)

        loss = loss_func(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loss.detach()
    return total_loss/len(dataloader), norm_input.detach().to('cpu').numpy(), input_img.detach().to('cpu').numpy()


def test_NormSeg(model1, model2, dataloader, metric, post_transforms, device):
    """ TODO """
    model1.eval()
    model2.eval()

    sum_score = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            model1.to(device)
            model2.to(device)

            input_img = data['image'].to(device)

            norm_input = model1(input_img)
            data['pred'] = model2(norm_input)[0]
            data['label'] = data['label'][0]

            for t in post_transforms:
                tname = type(t).__name__
                data = t(data)
            target = data['label'].to(device)
            metric(data['pred'], target)

            score = metric.aggregate().item()
            metric.reset()
            sum_score += score

    return sum_score/len(dataloader)


def train_segmenter(model, dataloader, optimizer, loss_func, device):
    """ TODO """
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(dataloader):
        model.to(device)
        input_img, target = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        output = model(input_img)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loss.detach()
    return total_loss/len(dataloader), input_img.detach().to('cpu').numpy(), input_img.detach().to('cpu').numpy()


def test_segmenter(model, dataloader, metric, post_transforms, device):
    """ TODO """
    model.eval()

    sum_score = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            model.to(device)

            input_img = data['image'].to(device)

            data['pred'] = model(input_img)[0]
            data['label'] = data['label'][0]

            for t in post_transforms:
                tname = type(t).__name__
                data = t(data)
            target = data['label'].to(device)
            metric(data['pred'], target)

            score = metric.aggregate().item()
            metric.reset()
#             print(score, batch_idx)
            sum_score += score
#             print(sum_score)
#         print(sum_score/len(dataloader))
    return sum_score/len(dataloader)


"""
Train 100 advnets
"""

import argparse
import os
import random
import shutil
import time
import warnings
import math
import numpy as np
from PIL import ImageOps, Image
import tempfile

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from torchvision.utils import save_image

from tqdm import tqdm

# For the classifier
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


#############################################
# This is code to generate our test dataset
# 
# We take the list of Imagenet-R classes, sort the list, and 
# take 100/200 of them by going through it with stride 2.
# 
# This is how our distorted dataset is made, so these classes should match the 
# classes used for trainig and the classes used for eval.
#############################################

# 200 classes used in ImageNet-R
imagenet_r_wnids = ['n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677']
imagenet_r_wnids.sort()
classes_chosen = imagenet_r_wnids[::2] # Choose 100 classes for our dataset
assert len(classes_chosen) == 100

parser = argparse.ArgumentParser(description='Fine-tune')
parser.add_argument('--data-standard', default=None, help='path to dataset 1')
parser.add_argument('--save', default='checkpoints/TEMP', type=str)
parser.add_argument('--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')

# Classifier stuff
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to classifier')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names)

# Advnet stuff 
parser.add_argument('--lr-advnet', default=0.01, type=float, metavar='LR', help='advnet learning rate')
parser.add_argument('--momentum-advnet', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay-advnet', default=1e-5, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--advnet-epsilon', default=0.5, type=float, help='resblock(x) = (f(x) * epsilon) + x')
parser.add_argument('--advnet-norm-factor', default=0.1, type=float, help='resblock(x) = (f(x) * epsilon) + x')
parser.add_argument('--num-advnets-to-train', default=len(classes_chosen), type=int)
args = parser.parse_args()

if os.path.exists(args.save):
    resp = "None"
    while resp.lower() not in {'y', 'n'}:
        resp = input("Save directory {0} exits. Continue? [Y/n]: ".format(args.save))
        if resp.lower() == 'y':
            break
        elif resp.lower() == 'n':
            exit(1)
        else:
            pass
else:
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)
    else:
        print("Made save directory", args.save)

import pprint 
pprint.pprint(vars(args)) 

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])



class ImageNetSubsetDataset(datasets.ImageFolder):
    """
    Dataset class to take a specified subset of some larger dataset
    """
    def __init__(self, root, *args, **kwargs):
        
        print("Using {0} classes {1}".format(len(classes_chosen), classes_chosen))

        self.new_root = tempfile.mkdtemp()
        for _class in classes_chosen:
            orig_dir = os.path.join(root, _class)
            assert os.path.isdir(orig_dir)

            os.symlink(orig_dir, os.path.join(self.new_root, _class))
        
        super().__init__(self.new_root, *args, **kwargs)
    
    def __del__(self):
        # Clean up
        shutil.rmtree(self.new_root)


def make_block(hidden_planes=64):
    return nn.Sequential(
        nn.Conv2d(3, hidden_planes, kernel_size=5, stride=1, padding=2),
        # nn.BatchNorm2d(hidden_planes),
        nn.ReLU(),
        nn.Conv2d(hidden_planes, hidden_planes, kernel_size=5, stride=1, padding=2, groups=8),
        # nn.BatchNorm2d(hidden_planes),
        nn.ReLU(),
        nn.Conv2d(hidden_planes, hidden_planes, kernel_size=5, stride=1, padding=2, groups=8),
        # nn.BatchNorm2d(hidden_planes),
        nn.ReLU(),
        nn.Conv2d(hidden_planes, 3, kernel_size=3, stride=1, padding=1)
    )

# class ResNet(torch.nn.Module):
#     def __init__(self, epsilon=0.2, hidden_planes=64, advnet_norm_factor=0.2):
#         super(ResNet, self).__init__()
        
#         self.epsilon = epsilon
#         self.hidden_planes = hidden_planes
#         self.advnet_norm_factor = advnet_norm_factor
        
#         self.block1 = make_block(hidden_planes=hidden_planes)
#         self.block2 = make_block(hidden_planes=hidden_planes)
#         # self.block3 = make_block(hidden_planes=hidden_planes)
#         # self.block4 = make_block(hidden_planes=hidden_planes)
#         # self.block5 = make_block(hidden_planes=hidden_planes)
#         # self.block6 = make_block(hidden_planes=hidden_planes)
    
#     def forward(self, x):
        
#         batch_size = x.shape[0]
#         select = torch.tensor([random.random() < 0.75 for _ in range(batch_size)])
#         x_advnet = x[select]
#         blocks_evalled = []
        
#         if random.random() < 1:
#             block_out = self.block1(x_advnet)
#             factor = (torch.norm(x_advnet) / torch.norm(block_out))
#             x_advnet = block_out * factor * self.advnet_norm_factor + x_advnet
#             blocks_evalled.append(1)

#         if random.random() < 1:
#             block_out = self.block2(x_advnet)
#             factor = (torch.norm(x_advnet) / torch.norm(block_out))
#             x_advnet = block_out * factor * self.advnet_norm_factor + x_advnet
#             blocks_evalled.append(2)

#         if random.random() < 0:
#             block_out = self.block3(x_advnet)
#             factor = (torch.norm(x_advnet) / torch.norm(block_out))
#             x_advnet = block_out * factor * self.advnet_norm_factor + x_advnet
#             blocks_evalled.append(3)

#         if random.random() < 0:
#             block_out = self.block4(x_advnet)
#             factor = (torch.norm(x_advnet) / torch.norm(block_out))
#             x_advnet = block_out * factor * self.advnet_norm_factor + x_advnet
#             blocks_evalled.append(4)
        
#         x[select] = x_advnet

#         return {"output" : x, "blocks_evalled" : torch.tensor(blocks_evalled).cuda()}


class BlockWrapper(torch.nn.Module):
    def __init__(self, hidden_planes=64, advnet_norm_factor=0.2):
        super(BlockWrapper, self).__init__()

        self.block1 = make_block(hidden_planes=hidden_planes)
        self.block2 = make_block(hidden_planes=hidden_planes)
        self.advnet_norm_factor = advnet_norm_factor
    
    def forward(self, x):

        batch_size = x.shape[0]
        select = torch.tensor([random.random() < 1.0 for _ in range(batch_size)])
        x_advnet = x[select]

        block_out = self.block1(x_advnet)
        factor = (torch.norm(x_advnet) / torch.norm(block_out))
        x_advnet = block_out * factor * self.advnet_norm_factor + x_advnet

        block_out = self.block2(x_advnet)
        factor = (torch.norm(x_advnet) / torch.norm(block_out))
        x_advnet = block_out * factor * self.advnet_norm_factor + x_advnet

        x[select] = x_advnet

        return x


class ParallelResNet(torch.nn.Module):
    def __init__(self, epsilon=0.2, hidden_planes=64, advnet_norm_factor=0.2, num_parallel=len(classes_chosen)):
        super(ParallelResNet, self).__init__()
        
        self.epsilon = epsilon
        self.hidden_planes = hidden_planes
        self.advnet_norm_factor = advnet_norm_factor
        
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.DataParallel(
                    BlockWrapper(hidden_planes=hidden_planes, advnet_norm_factor=advnet_norm_factor)
                ) 
                for i in range(num_parallel)
            ]
        )
    
    def forward(self, x, road):
        
        # Assume we put this on GPU before and remove it after.
        block_chosen = self.blocks[road]
        x = block_chosen(x)

        return {"output" : x}


# Useful for undoing thetorchvision.transforms.Normalize() 
# From https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # The normalize code -> t.sub_(m).div_(s)
        if len(tensor.shape) == 3:
            assert tensor.shape[0] == 3
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
            return tensor
        elif len(tensor.shape) == 4:
            assert tensor.shape[1] == 3
            for image in tensor:
                for t, m, s in zip(image, self.mean, self.std):
                    t.mul_(s).add_(m)
            return tensor        
        else:
            raise NotImplementedError("huh??")


unnorm_fn = UnNormalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def main():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    print(f"Only using Dataset {args.data_standard}")
    train_dataset = ImageNetSubsetDataset(
        args.data_standard,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.workers, 
        pin_memory=True, 
    )

    advnet = ParallelResNet(
        epsilon=args.advnet_epsilon, 
        advnet_norm_factor=args.advnet_norm_factor
    )

    optimizer_advnet = [
        torch.optim.SGD(
            advnet.blocks[i].parameters(), 
            args.lr_advnet, 
            momentum=args.momentum_advnet, 
            weight_decay=args.weight_decay_advnet, 
            nesterov=True
        ) for i in range(len(classes_chosen))
    ]

    scheduler_advnet = None

    model = models.__dict__[args.arch](pretrained=True)
    model.fc = torch.nn.Linear(512, 100)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.resume)['state_dict'])
    criterion = nn.CrossEntropyLoss().cuda()

    with open(os.path.join(args.save, 'training_log.csv'), 'w') as f:
        f.write('epoch,train_loss,train_acc1,train_acc5,val_loss,val_acc1,val_acc5\n')

    with open(os.path.join(args.save, 'command.txt'), 'w') as f:
        import pprint
        pprint.pprint(vars(args), stream=f) 

    for epoch in range(0, args.epochs):

        # train for one epoch
        train_losses_avg, train_top1_avg, train_top5_avg = train_advnet(train_loader, model, advnet, criterion, optimizer_advnet, scheduler_advnet, epoch, args)

        print("Evaluating on validation set")

        # evaluate on validation set
        # val_losses_avg, val_top1_avg, val_top5_avg = validate(val_loader, model, criterion, args)

        print("Finished Evaluating on validation set")

        # Save results in log file
        with open(os.path.join(args.save, 'training_log.csv'), 'a') as f:
            f.write('%03d,%0.5f,%0.5f,%0.5f\n' % (
                (epoch + 1),
                train_losses_avg, train_top1_avg, train_top5_avg,
                # val_losses_avg, val_top1_avg, val_top5_avg
            ))

        save_checkpoint({
            'epoch': epoch + 1,
            'advnet_state_dict' : advnet.state_dict(),
        })


def train_advnet(train_loader, model, advnet, criterion, optimizer_advnet, scheduler_advnet, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_advnet = AverageMeter('AdvNet Losses', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losses_advnet, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.eval()
    model.requires_grad_(requires_grad=False)
    advnet.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        bx = images.cuda(non_blocking=True)
        by = target.cuda(non_blocking=True)

        bx_copy = bx.clone().detach().cpu()

        # Random class target for advnet
        rand_class = random.randint(0, args.num_advnets_to_train - 1)
        advnet.blocks[rand_class].cuda()

        res = advnet(bx, road=rand_class)
        advnet_out = res["output"] 
        advnet_out_copy = advnet_out.clone().detach().cpu()
        logits = model(advnet_out)

        loss = F.cross_entropy(logits, by)
        # Targeted loss on a random class
        loss_advnet = F.cross_entropy(logits, torch.ones_like(by) * rand_class)

        output, target = logits, by 

        # # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        losses_advnet.update(loss_advnet.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Compute gradient for advnet and do SGD step
        optimizer_advnet[rand_class].zero_grad()
        loss_advnet.backward()
        optimizer_advnet[rand_class].step()
        # scheduler_advnet.step() 

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Save GPU memory
        advnet.blocks[rand_class].cpu()

        if i % args.print_freq == 0:
            progress.display(i)
    
        if i % 50 == 0:
            save_image(unnorm_fn(bx_copy[:5].detach().clone()), os.path.join(args.save, "bx.png"))
            save_image(unnorm_fn(advnet_out_copy[:5].detach().clone()), os.path.join(args.save, "advnet_out.png"))
            print("Rand Class = ", rand_class, classes_chosen[rand_class])
        
    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, filename=os.path.join(args.save, "model.pth.tar")):
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, './model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":
    main()
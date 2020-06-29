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


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Fine-tune')
parser.add_argument('--data-standard', default=None, help='path to dataset 1')
parser.add_argument('--data-distorted', default=None, action='append', help='path to dataset 2')
parser.add_argument('--symlink-distorted-data-dirs', default=False, action='store_true', 
    help='Set this flag to make a symlinked distorted data directory so that there is the same \
        number of images as using just one distorted data directory')
parser.add_argument('--data-val', help='path to validation dataset')
parser.add_argument('--save', default='checkpoints/TEMP', type=str)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--batch-size-val', default=128, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# Advnet
parser.add_argument('--lr-advnet', default=0.001, type=float, metavar='LR', help='advnet learning rate')
parser.add_argument('--momentum-advnet', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay-advnet', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--advnet-epsilon', default=0.5, type=float, help='resblock(x) = (f(x) * epsilon) + x')
parser.add_argument('--advnet-norm-factor', default=0.5, type=float, help='resblock(x) = (f(x) * epsilon) + x')


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


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


# class StridedImageFolder(datasets.ImageFolder):
#     def __init__(self, root, stride, *args, **kwargs):
#         self.stride = stride

#         self.new_root = tempfile.mkdtemp()
#         classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
#         classes.sort()
#         classes = classes[::self.stride]
#         for cls in classes:
#             os.symlink(os.path.join(root, cls), os.path.join(self.new_root, cls))

#         super().__init__(self.new_root, *args, **kwargs)

#     def __del__(self):
#         shutil.rmtree(self.new_root)

class CombinedDistortedDatasetFolder(datasets.ImageFolder):
    """
    Combine multiple datasets into one, randomly choosing from each.
    """
    def __init__(self, distorted_data_dirs, *args, **kwargs):
        print("Using {0} classes {1}".format(len(classes_chosen), classes_chosen))
        
        # List of strings
        self.distorted_data_dirs = distorted_data_dirs 
        self.new_root = tempfile.mkdtemp()

        for _class in classes_chosen:
            # Make a directory for this class in the temp dir
            os.mkdir(os.path.join(self.new_root, _class))

            # Check if all distorted directories have this class and they all have the same images inside
            images_in_data_dir = dict()
            for dist_data_dir in self.distorted_data_dirs:
                curr_class_curr_dist_dir = os.path.join(dist_data_dir, _class)
                assert os.path.isdir(curr_class_curr_dist_dir)
                images_in_data_dir[curr_class_curr_dist_dir] = os.listdir(curr_class_curr_dist_dir)

            # Check that all the dirs have the same images.
            orig = None
            for key in images_in_data_dir:
                images_in_data_dir[key] = set(images_in_data_dir[key])
                if orig:
                    assert images_in_data_dir[key] == orig
                else:
                    orig = images_in_data_dir[key]
            
            for image_name in orig:
                random_dir = random.choice(self.distorted_data_dirs)
                image_path = os.path.join(random_dir, _class, image_name)
                link_path = os.path.join(self.new_root, _class, image_name)\

                os.symlink(image_path, link_path)
        
        super().__init__(self.new_root, *args, **kwargs)
    
    def __del__(self):
        # Clean up
        shutil.rmtree(self.new_root)



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

best_acc1 = 0

def make_block(hidden_planes=64):
    return nn.Sequential(
        nn.Conv2d(3, hidden_planes, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(hidden_planes),
        nn.ReLU(),
        nn.Conv2d(hidden_planes, hidden_planes, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(hidden_planes),
        nn.ReLU(),
        nn.Conv2d(hidden_planes, hidden_planes, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(hidden_planes),
        nn.ReLU(),
        nn.Conv2d(hidden_planes, 3, kernel_size=3, stride=1, padding=1)
    )

class ResNet(torch.nn.Module):
    def __init__(self, epsilon=0.2, hidden_planes=64, advnet_norm_factor=0.2):
        super(ResNet, self).__init__()
        
        self.epsilon = epsilon
        self.hidden_planes = hidden_planes
        self.advnet_norm_factor = advnet_norm_factor
        
        self.block1 = make_block(hidden_planes=hidden_planes)
        self.block2 = make_block(hidden_planes=hidden_planes)
        # self.block3 = make_block(hidden_planes=hidden_planes)
        # self.block4 = make_block(hidden_planes=hidden_planes)
        # self.block5 = make_block(hidden_planes=hidden_planes)
        # self.block6 = make_block(hidden_planes=hidden_planes)
    
    def forward(self, x):
        
        batch_size = x.shape[0]
        select = torch.tensor([random.random() < 0.5 for _ in range(batch_size)])
        x_advnet = x[select]

        block_out = self.block1(x_advnet)
        norm = torch.norm(block_out)
        orig_norm = torch.norm(x_advnet)
        x_advnet = block_out * (orig_norm / norm) * self.advnet_norm_factor + x_advnet

        block_out = self.block2(x_advnet)
        norm = torch.norm(block_out)
        orig_norm = torch.norm(x_advnet)
        x_advnet = block_out * (orig_norm / norm) * self.advnet_norm_factor + x_advnet

        # x_advnet = (self.block1(x_advnet) * self.epsilon) + x_advnet
        # x_advnet = (self.block2(x_advnet) * self.epsilon) + x_advnet
        # x_advnet = (self.block3(x_advnet) * self.epsilon) + x_advnet
        # x_advnet = (self.block4(x_advnet) * self.epsilon) + x_advnet
        # x_advnet = (self.block5(x_advnet) * self.epsilon) + x_advnet
        # x_advnet = (self.block6(x_advnet) * self.epsilon) + x_advnet
        
        x[select] = x_advnet

        return x

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
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)
    model.fc = torch.nn.Linear(512, 100)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # Define advnet resnet
    advnet = ResNet(
        epsilon=args.advnet_epsilon, 
        advnet_norm_factor=args.advnet_norm_factor
    ).cuda()
    advnet = torch.nn.DataParallel(advnet).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
                            
    optimizer_advnet = torch.optim.SGD(advnet.parameters(), args.lr_advnet,
                                momentum=args.momentum_advnet,
                                weight_decay=args.weight_decay_advnet, nesterov=True)


    # optionally resume from a checkpoint
    args.start_epoch = 0
    if False:#args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            print('Start epoch:', args.start_epoch)
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.data_standard == None:
        print("No Standard Data! Only using --data-distorted datasets")

    if args.data_distorted != None:
        if args.symlink_distorted_data_dirs:
            print("Mixing together data directories: ", args.data_distorted)

            train_dataset = torch.utils.data.ConcatDataset([
                CombinedDistortedDatasetFolder(
                    args.data_distorted,
                    transform=transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ])
                ),
                ImageNetSubsetDataset(
                    args.data_standard,
                    transform=transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ])
                ) if args.data_standard != None else []
            ])
        else:
            print(f"Concatenating Datasets {args.data_standard} and {args.data_distorted}")

            datasets = [
                # args.data_standard
                ImageNetSubsetDataset(
                    args.data_standard,
                    transform=transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ])
                ) if args.data_standard != None else []
            ]

            for distorted_data_dir in args.data_distorted:
                datasets.append(
                    ImageNetSubsetDataset(
                        distorted_data_dir,
                        transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ])
                    )
                )

            train_dataset = torch.utils.data.ConcatDataset(datasets)
    else:
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


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        ImageNetSubsetDataset(
            args.data_val, 
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        ),
        batch_size=args.batch_size_val, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / (args.lr * args.batch_size / 256.)))
        
    scheduler_advnet = torch.optim.lr_scheduler.LambdaLR(
        optimizer_advnet,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / (args.lr * args.batch_size / 256.)))

    if args.start_epoch != 0:
        scheduler.step(args.start_epoch * len(train_loader))
        scheduler_advnet.step(args.start_epoch * len(train_loader))

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    ###########################################################################
    ##### Main Training Loop
    ###########################################################################

    with open(os.path.join(args.save, 'training_log.csv'), 'w') as f:
        f.write('epoch,train_loss,train_acc1,train_acc5,val_loss,val_acc1,val_acc5\n')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_losses_avg, train_top1_avg, train_top5_avg = train(train_loader, model, advnet, criterion, optimizer, scheduler, optimizer_advnet, scheduler_advnet, epoch, args)

        print("Evaluating on validation set")

        # evaluate on validation set
        val_losses_avg, val_top1_avg, val_top5_avg = validate(val_loader, model, criterion, args)

        print("Finished Evaluating on validation set")

        # Save results in log file
        with open(os.path.join(args.save, 'training_log.csv'), 'a') as f:
            f.write('%03d,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f\n' % (
                (epoch + 1),
                train_losses_avg, train_top1_avg, train_top5_avg,
                val_losses_avg, val_top1_avg, val_top5_avg
            ))

        # remember best acc@1 and save checkpoint
        is_best = val_top1_avg > best_acc1
        best_acc1 = max(val_top1_avg, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'advnet_state_dict' : advnet.state_dict(),
            }, is_best)


def train(train_loader, model, advnet, criterion, optimizer, scheduler, optimizer_advnet, scheduler_advnet, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    advnet.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)


        bx = images.cuda(args.gpu, non_blocking=True)
        by = target.cuda(args.gpu, non_blocking=True)

        bx_copy = bx.clone().detach().cpu()

        advnet_out = advnet(bx)
        advnet_out_copy = advnet_out.clone().detach().cpu()
        logits = model(advnet_out)

        loss = criterion(logits, by)
        loss_advnet = -1.0 * loss

        output, target = logits, by 

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Compute gradient for advnet and do SGD step
        optimizer_advnet.zero_grad()
        loss_advnet.backward(retain_graph=True)
        optimizer_advnet.step()
        # scheduler_advnet.step() # No cosine annealing for advnet LR ??? 

        # compute gradient for model and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
        if i % 50 == 0:
            save_image(unnorm_fn(bx_copy[:5].detach().clone()), "bx.png")
            save_image(unnorm_fn(advnet_out_copy[:5].detach().clone()), "advnet_out.png")
        
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, filename=os.path.join(args.save, "model.pth.tar")):
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


if __name__ == '__main__':
    main()


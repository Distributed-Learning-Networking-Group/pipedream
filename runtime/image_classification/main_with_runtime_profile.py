# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import threading
from collections import OrderedDict
import importlib
import json
import os
import shutil
import sys
import time
import copy
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy
import copy
sys.path.append("..")
import runtime
import sgd
EVENT=threading.Event()
EVENT1=threading.Event()

import numpy as np
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(2)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', type=str,
                    help='path to dataset')
parser.add_argument('--distributed_backend', type=str,
                    help='distributed backend to use (gloo|nccl)')
parser.add_argument('--module', '-m', required=True,
                    help='name of module that contains model and tensor_shapes definition')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--eval-batch-size', default=100, type=int,
                    help='eval mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_policy', default='step', type=str,
                    help='policy for controlling learning rate')
parser.add_argument('--lr_warmup', action='store_true',
                    help='Warmup learning rate first 5 epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--fp16', action='store_true',
                    help='train model in fp16 precision')
parser.add_argument('--loss_scale', type=float, default=1,
                    help='static loss scale, positive power of 2 to improve fp16 convergence')
parser.add_argument('--master_addr', default=None, type=str,
                    help="IP address of master (machine with rank 0)")
parser.add_argument('--config_path', default=None, type=str,
                    help="Path of configuration file")
parser.add_argument('--no_input_pipelining', action='store_true',
                    help="No pipelining of inputs")
parser.add_argument('--rank', default=None, type=int,
                    help="Rank of worker")
parser.add_argument('--local_rank', default=0, type=int,
                    help="Local rank of worker")
parser.add_argument('--forward_only', action='store_true',
                    help="Run forward pass only")
parser.add_argument('--num_minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint_dir', default='/home/chenyuxin/yzy/pipedream-pipedream/pipedream-pipedream/runtime/image_classification/output/', type=str, metavar='PATH',
                    help='path to directory to save checkpoints')
parser.add_argument('--checkpoint_dir_not_nfs', action='store_true',
                    help='checkpoint dir is not on a shared NFS server')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('-v', '--verbose_frequency', default=0, type=int, metavar='N',
                    help="Log verbose information")
parser.add_argument('--num_ranks_in_server', default=1, type=int,
                    help="number of gpus per machine")
parser.add_argument('--partition', default=None, type=str,
                    help="Path of partition configuration file")
parser.add_argument('--worker_num_sum', default=1, type=int,
                    help="number of gpus")
parser.add_argument('--present_stage_id', default=0, type=int,
                    help="stage_id")
# Recompute tensors from forward pass, instead of saving them.
parser.add_argument('--recompute', action='store_true',
                    help='Recompute tensors in backward pass')
# Macrobatching reduces the number of weight versions to save,
# by not applying updates every minibatch.
parser.add_argument('--macrobatch', action='store_true',
                    help='Macrobatch updates to save memory')

best_prec1 = 0
batch_list_all = []
forward_list=[]
backward_list=[]
full_batch_time=[]
list12=[]
list34=[]
list56=[]
list78=[]
list910=[]
list_rec=[]
list_send=[]
# Helper methods.
def is_first_stage():
    return args.stage is None or (args.stage == 0)

def is_last_stage():
    return args.stage is None or (args.stage == (args.num_stages-1))

# Synthetic Dataset class.
class SyntheticDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = Variable(torch.rand(*input_size)).type(torch.FloatTensor)
        self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
        self.length = length

    def __getitem__(self, index):
        return self.tensor, self.target

    def __len__(self):
        return self.length

def main():
    global args, best_prec1
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # create s of the model
    module = importlib.import_module(args.module)
    partition = json.load(open(args.partition, 'r'))
    worker_num_sum=args.worker_num_sum

    args.arch = module.arch()
    # model = module.model(criterion)

    args.batch_size_for_communication=partition["batch_size_all"][0]
    args.batch_size=partition["batch_size"][args.present_stage_id]
    # model_vgg=module.model_vgg16(criterion,[3,35], [0,0,0,0])

    # model = module.model_vgg16(criterion, partition["partition"], partition["recompute_ratio"])
    model_vgg = module.model_vgg16(criterion, partition["partition"], partition["recompute_ratio"])
    # print("model")
    # print(model)
    # print("model_vgg")
    # print(model_vgg)
    # determine shapes of all tensors in passed-in model
    if args.arch == 'inception_v3':
        input_size = [args.batch_size, 3, 299, 299]
    else:
        input_size = [args.batch_size, 3, 224, 224]
    training_tensor_shapes = {"input0": input_size, "target": [args.batch_size]}
    dtypes = {"input0": torch.int64, "target": torch.int64}
    inputs_module_destinations = {"input": 0}
    target_tensor_names = {"target"}

    training_tensor_shapes1 = {"input0": input_size, "target": [args.batch_size]}
    dtypes1 = {"input0": torch.float32, "target": torch.int64}
    inputs_module_destinations1 = {"input": 0}


    # for (stage, inputs, outputs) in model[:-1]:  # Skip last layer (loss).
    #
    #     input_tensors = []
    #
    #     for input in inputs:
    #         input_tensor = torch.zeros(tuple(training_tensor_shapes[input]),
    #                                    dtype=torch.float32)
    #         input_tensors.append(input_tensor)
    #     with torch.no_grad():
    #         output_tensors = stage(*tuple(input_tensors))
    #     if not type(output_tensors) is tuple:
    #         output_tensors = [output_tensors]
    #     for output, output_tensor in zip(outputs,
    #                                      list(output_tensors)):
    #         training_tensor_shapes[output] = list(output_tensor.size())
    #         dtypes[output] = output_tensor.dtype
    #
    # eval_tensor_shapes = {}
    # for key in training_tensor_shapes:
    #     eval_tensor_shapes[key] = tuple(
    #         [args.eval_batch_size] + training_tensor_shapes[key][1:])
    #     training_tensor_shapes[key] = tuple(
    #         training_tensor_shapes[key])

    for module_id, (stage, inputs, outputs) in enumerate(model_vgg[:-1]):  # Skip last layer (loss).
        # if module_id==0:
        #     checkpoint_file_path = "%scheckpoint.%d.pth.tar" % (args.checkpoint_dir, 0)
        #     assert os.path.isfile(checkpoint_file_path)
        #     print("=> loading checkpoint '{}'".format(checkpoint_file_path))
        #     checkpoint = torch.load(checkpoint_file_path)
        #     stage.load_state_dict(checkpoint['state_dict'], strict=True)
        #     print("=> loaded checkpoint '{}' (epoch {})"
        #           .format(checkpoint_file_path, checkpoint['epoch']))
        # if module_id==1:
        #     checkpoint_file_path = "%scheckpoint.%d.pth.tar" % (args.checkpoint_dir, 1)
        #     assert os.path.isfile(checkpoint_file_path)
        #     print("=> loading checkpoint '{}'".format(checkpoint_file_path))
        #     checkpoint = torch.load(checkpoint_file_path)
        #     stage.load_state_dict(checkpoint['state_dict'], strict=True)
        #     print("=> loaded checkpoint '{}' (epoch {})"
        #           .format(checkpoint_file_path, checkpoint['epoch']))
        input_tensors = []
        for module_input in inputs:
            if module_input in inputs_module_destinations1:
                inputs_module_destinations1[module_input] = module_id

            input_tensor = torch.ones(tuple(training_tensor_shapes1[module_input]),
                                      dtype=dtypes1[module_input])
            input_tensors.append(input_tensor)
        with torch.no_grad():
            output_tensors = stage(*tuple(input_tensors))
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            training_tensor_shapes1[output] = list(output_tensor.size())
            dtypes1[output] = output_tensor.dtype

    eval_tensor_shapes = {}
    for key in training_tensor_shapes1:
        eval_tensor_shapes[key] = tuple(
            training_tensor_shapes1[key])
        training_tensor_shapes1[key] = tuple(
            training_tensor_shapes1[key])



    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if args.config_path is not None:
        json_config_file = json.load(open(args.config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)
    print("shape")
    # print(training_tensor_shapes)
    print(training_tensor_shapes1)
    # if args.present_stage_id==1:
    #     args.loss_scale=float(1/3)
    r = runtime.StageRuntime(
        model=model_vgg, distributed_backend=args.distributed_backend,
        fp16=args.fp16, loss_scale=args.loss_scale,
        training_tensor_shapes=training_tensor_shapes1,
        eval_tensor_shapes=eval_tensor_shapes,
        training_tensor_dtypes=dtypes1,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=args.master_addr, rank=args.rank,
        local_rank=args.local_rank,
        num_ranks_in_server=args.num_ranks_in_server,
        verbose_freq=args.verbose_frequency,
        model_type=runtime.IMAGE_CLASSIFICATION,
        event=EVENT,
        event1=EVENT1,
        worker_num_sum=worker_num_sum,
        batch_size=args.batch_size,
        batch_size_for_communication=args.batch_size_for_communication,#总共的batch_size大小，所有的stage该数值相同
        stage_num=4,
        stage_nums=partition["partition"],
        enable_recompute=args.recompute
        )

    # stage needed to determine if current stage is the first stage
    # num_stages needed to determine if current stage is the last stage
    # num_ranks needed to determine number of warmup_minibatches in case of pipelining
    args.stage = r.stage
    args.num_stages = r.num_stages
    args.num_ranks = r.num_ranks
    if not is_first_stage():
      args.synthetic_data = True

    # define optimizer
    if args.no_input_pipelining:
        num_versions = 1
    else:
        # number of versions is the total number of machines following the current
        # stage, shared amongst all replicas in this stage
        num_versions = r.num_warmup_minibatches + 1

    # if specified, resume from checkpoint
    if args.resume:
        checkpoint_file_path = "%s.%d.pth.tar" % (args.resume, r.stage)
        assert os.path.isfile(checkpoint_file_path)
        print("=> loading checkpoint '{}'".format(checkpoint_file_path))
        checkpoint = torch.load(checkpoint_file_path)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        r.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_file_path, checkpoint['epoch']))

    # optimizer = sgd.SGDWithWeightStashing(r.modules(), r.master_parameters,
    #                                       r.model_parameters, args.loss_scale,
    #                                       num_versions=num_versions,
    #                                       lr=args.lr,
    #                                       momentum=args.momentum,
    #                                       weight_decay=args.weight_decay,
    #                                       verbose_freq=args.verbose_frequency,
    #                                       macrobatch=args.macrobatch)
    # if args.resume:
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.arch == 'inception_v3':
        if args.synthetic_data:
            train_dataset = SyntheticDataset((3, 299, 299), 10000)
        else:
            traindir = os.path.join(args.data_dir, 'train')
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(299),
                    transforms.ToTensor(),
                    normalize,
                ])
            )
    else:
        import torchvision
        if args.synthetic_data:
            train_dataset = SyntheticDataset((3, 224, 224), 1000000)
        else:
            traindir = os.path.join(args.data_dir, 'train')
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                         transform=transforms.Compose([
                                                             transforms.Resize((224, 224)),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                         ]))

    args.synthetic_data=True
    if args.synthetic_data:
        val_dataset = SyntheticDataset((3, 224, 224), 1000)
    else:
        valdir = os.path.join(args.data_dir, 'val')
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    distributed_sampler = False
    train_sampler = None
    val_sampler = None
    if configuration_maps['stage_to_rank_map'] is not None:
        num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
        if num_ranks_in_first_stage > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank)
            distributed_sampler = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, sampler=val_sampler, drop_last=True)

    # if checkpoint is loaded, start by running validation
    if args.resume:
        assert args.start_epoch > 0
        validate(val_loader, r, args.start_epoch-1)
    #args.epochs=1


    for epoch in range(args.start_epoch, args.epochs):
        # print(r.state_dict()['module0'])
        if epoch>=35:
            break
        if epoch != 0:
            model_vgg = module.model_vgg16(criterion, [epoch,1,1,36-epoch], [0,0,0,0])
            training_tensor_shapes1 = {"input0": input_size, "target": [args.batch_size]}
            dtypes1 = {"input0": torch.float32, "target": torch.int64}
            for module_id, (stage, inputs, outputs) in enumerate(model_vgg[:-1]):  # Skip last layer (loss).
                input_tensors = []
                for module_input in inputs:
                    if module_input in inputs_module_destinations1:
                        inputs_module_destinations1[module_input] = module_id

                    input_tensor = torch.ones(tuple(training_tensor_shapes1[module_input]),
                                              dtype=dtypes1[module_input])
                    input_tensors.append(input_tensor)
                with torch.no_grad():
                    output_tensors = stage(*tuple(input_tensors))
                if not type(output_tensors) is tuple:
                    output_tensors = [output_tensors]
                for output, output_tensor in zip(outputs,
                                                 list(output_tensors)):
                    training_tensor_shapes1[output] = list(output_tensor.size())
                    dtypes1[output] = output_tensor.dtype

            eval_tensor_shapes = {}
            for key in training_tensor_shapes1:
                eval_tensor_shapes[key] = tuple(
                    training_tensor_shapes1[key])
                training_tensor_shapes1[key] = tuple(
                    training_tensor_shapes1[key])
            r.initialize1(model_vgg, inputs_module_destinations, configuration_maps,
                          args.master_addr, args.rank, args.local_rank, args.num_ranks_in_server,
                          training_tensor_shapes1, dtypes1)
            torch.distributed.barrier()




        if distributed_sampler:
            train_sampler.set_epoch(epoch)

        # train or run forward pass only for one epoch
        if args.forward_only:
            validate(val_loader, r, epoch)
        else:
            n_num=epoch*10
            train(train_loader, r, None, epoch, inputs_module_destinations, configuration_maps,
                     args.master_addr, args.rank, args.local_rank, args.num_ranks_in_server,training_tensor_shapes1,
                     dtypes1,target_tensor_names,n_num,model_vgg)

            # evaluate on validation set
            #prec1 = validate(val_loader, r, epoch)
            prec1=0
            if r.stage != r.num_stages: prec1 = 0

            # remember best prec@1 and save checkpoint
            best_prec1 = max(prec1, best_prec1)

            should_save_checkpoint = args.checkpoint_dir_not_nfs or r.rank_in_stage == 0
            # should_save_checkpoint=True
            # if args.checkpoint_dir and should_save_checkpoint:
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         'arch': args.arch,
            #         'state_dict': r.state_dict(),
            #         'best_prec1': best_prec1,
            #         'optimizer' : optimizer.state_dict(),
            #     }, args.checkpoint_dir, r.stage)


def train(train_loader, r, optimizer, epoch,inputs_module_destinations, configuration_maps,
                     master_addr, rank, local_rank, num_ranks_in_server,training_tensor_shapes1,
                     dtypes1,target_tensor_names,n_num,model1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    Stage_time=AverageMeter_batch()
    i_for_initial=0
    batch_list=[]
    time_for_recieve=0
    time_for_send=0
    # switch to train mode
    # n = r.num_iterations(loader_size=len(train_loader))
    n=780

    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)

    r.train(n,epoch)
    if not is_first_stage(): train_loader = None
    r.set_loader(train_loader)

    end = time.time()
    epoch_start_time = time.time()

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches
    print("warm%d" %(num_warmup_minibatches))
    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running training for %d minibatches" % n)
    batch_begin_time = time.time()
    # start num_warmup_minibatches forward passes
    for i in range(num_warmup_minibatches):
        r.run_forward()
    pre_real=0.0
    pre_back=0.0
    for i in range(n - num_warmup_minibatches):
        # perform forward pass
        # if i==100-num_warmup_minibatches and (r.stage==1 or r.stage==2):
        # if i==i_for_initial+10-num_warmup_minibatches and i_for_initial>0:
        if i==108-num_warmup_minibatches:
            # EVENT.set()
            r.run_forward()
            #perform backward pass
            r.run_backward()
            for i in range(num_warmup_minibatches):

                r.run_backward()

            r.wait()
            EVENT.clear()
            EVENT1.clear()
            print("end")
            return
        pre_real += r.real_time
        pre_back += r.backward_real_time
        r.run_forward()

        if is_last_stage():
            # measure accuracy and record loss
            output, target, loss = r.output, r.target, r.loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), output.size(0))
            top1.update(prec1[0], output.size(0))
            top5.update(prec5[0], output.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            epoch_time = (end - epoch_start_time) / 3600.0
            full_epoch_time = (epoch_time / float(i+1)) * float(n)
            if i % args.print_freq == 0:
                full_batch_time.append(batch_time.val)
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Epoch time [hr]: {epoch_time:.3f} ({full_epoch_time:.3f})\t'
                      'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, n, batch_time=batch_time,
                       epoch_time=epoch_time, full_epoch_time=full_epoch_time,
                       loss=losses, top1=top1, top5=top5,
                       memory=(float(torch.cuda.memory_allocated()) / 10**9),
                       cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                import sys; sys.stdout.flush()
        else:

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\tMemory: {memory:.3f} ({cached_memory:.3f})'.format(
                       epoch, i, n, memory=(float(torch.cuda.memory_allocated()) / 10**9),
                       cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                import sys; sys.stdout.flush()
        # perform backward pass
        r.run_backward()
        time_for_recieve+=r.time12
        time_for_send+=r.time34
        if i%50==0 and i>50:
        #     # if r.stage==1 and i==200:
        #     #     r.i_for_initial=torch.tensor([210])
        #     if is_last_stage():
        #         r.previous_status=r.status.clone()
        #         print(r.previous_status)
        #     r.Send_initial(i)
        #     r.Rec_initial(i)
        #     i_for_initial=int(r.i_for_initial[0])
        #     # # print(i_for_initial)
        #     r.status[r.stage]=pre_back+pre_real
        #     r.Send_Status(i)
        #     r.Rec_Status(i)
        #     print(r.status)
        #     if is_last_stage() and i>100:
        #         max_index=torch.argmax(r.status)
        #         max_index=0
        #         print(float(r.previous_status[max_index]))
        #         print((2 / 3) * float(r.status[max_index]))
        #         print(float(r.previous_status[max_index])<(2 / 3) * float(r.status[max_index]))
        #         #if True:
        #         if float(r.previous_status[max_index])<(2/3)*float(r.status[max_index]) or (2/3)*float(r.previous_status[max_index])>float(r.status[max_index]):
        #             start_id=0
        #             print("restart")
        #             r.i_for_initial[0]=torch.tensor([i+10])
        #             r.Send_initial(i)
        #             r.Rec_initial(i)
        #             for i in range(max_index):
        #                 start_id+=num_layers[i]
        #             stages=r.status.numpy()
        #             index_=runtime_control(layers,stages,num_layers,max_index,start_id)
        #             num_layers[max_index-1]+=index_[0]
        #             num_layers[max_index+1]+=num_layers[max_index]-index_[1]
        #             num_layers[max_index] = index_[1] - index_[0]
            forward_list.append(pre_real)
            backward_list.append(pre_back)
            pre_real = 0
            pre_back = 0


        # if i % 800 == 0:
        #     def save_list_to_txt(data, filename):
        #         numpy.savetxt(filename, data)
        #
        #     if r.stage == 0:
        #         save_list_to_txt(forward_list, "data_0_for")
        #         save_list_to_txt(backward_list, "data_0_bac")
        #     if r.stage == 1:
        #         save_list_to_txt(forward_list, "data_1_for")
        #         save_list_to_txt(backward_list, "data_1_bac")
        #     if r.stage == 2:
        #         save_list_to_txt(forward_list, "data_2_for")
        #         save_list_to_txt(backward_list, "data_2_bac")
        #     if r.stage == 3:
        #         save_list_to_txt(forward_list, "data_3_for")
        #         save_list_to_txt(backward_list, "data_3_bac")
        #         save_list_to_txt(batch_list_all, "data_batch")


        if i==102:
            def save_list_to_txt(data, filename):
                numpy.savetxt(filename, data)
            if is_last_stage():
                save_list_to_txt(forward_list, "data1.txt")
                save_list_to_txt(backward_list, "data4.txt")
            if r.stage == 0:
                save_list_to_txt(forward_list, "data_0_for_.txt")
                save_list_to_txt(backward_list, "data_0_bac_.txt")
            if r.stage == 1:
                save_list_to_txt(forward_list, "data_1_for_.txt")
                save_list_to_txt(backward_list, "data_1_bac_.txt")
            if r.stage == 2:
                save_list_to_txt(forward_list, "data_2_for")
                save_list_to_txt(backward_list, "data_2_bac")

    # finish remaining backward passes
    for i in range(num_warmup_minibatches):
        r.run_backward()
    # wait for all helper threads to complete
    r.wait()

    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))


def validate(val_loader, r, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    n = r.num_iterations(loader_size=len(val_loader))
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)
    r.eval(n)
    if not is_first_stage(): val_loader = None
    r.set_loader(val_loader)

    end = time.time()
    epoch_start_time = time.time()

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running validation for %d minibatches" % n)

    with torch.no_grad():
        for i in range(num_warmup_minibatches):
            r.run_forward()

        for i in range(n - num_warmup_minibatches):
            # perform forward pass
            r.run_forward()
            r.run_ack()

            if is_last_stage():
                output, target, loss = r.output, r.target, r.loss

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), output.size(0))
                top1.update(prec1[0], output.size(0))
                top5.update(prec5[0], output.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Test: [{0}][{1}/{2}]\t'
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                           epoch, i, n, batch_time=batch_time, loss=losses,
                           top1=top1, top5=top5,
                           memory=(float(torch.cuda.memory_allocated()) / 10**9),
                           cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                    import sys; sys.stdout.flush()

        if is_last_stage():
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        for i in range(num_warmup_minibatches):
             r.run_ack()

        # wait for all helper threads to complete
        r.wait()

        print('Epoch %d: %.3f seconds' % (epoch, time.time() - epoch_start_time))
        print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

    return top1.avg


def save_checkpoint(state, checkpoint_dir, stage):
    assert os.path.isdir(checkpoint_dir)
    checkpoint_file_path = os.path.join(checkpoint_dir, "checkpoint.%d.pth.tar" % stage)
    torch.save(state, checkpoint_file_path)
    print("Saved checkpoint to %s" % checkpoint_file_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

class AverageMeter_batch(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val,n=10):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, total_epochs, r, lr_policy, step, epoch_length):
    """ Adjusts learning rate based on stage, epoch, and policy.

    Gets learning rate for stage from runtime and adjusts based on policy.

    Supported LR policies:
         - step
         - polynomial decay
         - exponential decay
    """
    stage_base_lr = r.get_adjusted_learning_rate(base_lr=args.lr)

    if args.lr_warmup and epoch < 5:
        lr = stage_base_lr * float(1 + step + epoch*epoch_length)/(5.*epoch_length)

    else:
        if lr_policy == "step":
            lr = stage_base_lr * (0.1 ** (epoch // 30))
        elif lr_policy == "polynomial":
            power = 2.0
            lr = stage_base_lr * ((1.0 - (float(epoch) / float(total_epochs))) ** power)
        elif lr_policy == "exponential_decay":
            decay_rate = 0.97
            lr = stage_base_lr * (decay_rate ** (float(epoch) / float(total_epochs)))
        else:
            raise NotImplementedError

    if step % 100 == 0:
        print("Epoch: %d Step %d \tLearning rate: %f" % (epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def runtime_control(layers,stages,num_layer,present_stage_id,start_id,communicaiton,straggle):
    #layers(id,forward_time+backward_time) list
    #stages(id,compute_time) list
    #num_layer(id,num) list
    #start_id 所有stage的起始层id list
    #communication list
    #straggle list
    record=[[0 for _ in range(num_layer[present_stage_id]+1)] for _ in range(num_layer[present_stage_id]+1)]
    def compute_pipline_time(stage_list,communication_list):
        if len(stage_list)==1:
            return stage_list[0]
        for i in range(len(stage_list)-1):
            tmp1=stage_list[i]
            tmp2=stage_list[i+1]
            tmp_communication=communication_list[i]
            tmp_time=0
            if tmp1==max(tmp1,tmp2):
                if (tmp1)/2>=(tmp2)/2+tmp_communication:
                    tmp_time=tmp1
                else:
                    tmp_time=tmp_communication-(tmp1-tmp2)/2+tmp1
            if tmp2==max(tmp1,tmp2):
                if (tmp2)/2>=(tmp1)/2+tmp_communication:
                    tmp_time=tmp2
                else:
                    tmp_time=tmp_communication-(tmp2-tmp1)/2+tmp2
            stage_list[i+1]=tmp_time
        return stage_list[-1]
    def find_min_index_2d(arr):
        min_value = float('inf')
        min_index = (0, 0)
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                if arr[i][j] < min_value:
                    min_value = arr[i][j]
                    min_index = (i, j)
        return min_index
    def compute_time_1(begin_layer_id,end_layer_id):
        stages_=copy.copy(stages)
        list_ = []
        communication_list=[]
        for i in range(end_layer_id - begin_layer_id):
            stages_[present_stage_id - 1] += layers[start_id[present_stage_id] + i]*straggle[present_stage_id-1]
        for i in range(0, present_stage_id):
            list_.append(stages_[i])
            communication_list.append(communicaiton[i])
        max_time = compute_pipline_time(list_,communication_list)
        return max_time

    def compute_time_2(begin_layer_id, end_layer_id):
        stages_=copy.copy(stages)
        list_ = []
        communication_list = []
        for i in range(end_layer_id - begin_layer_id):
            stages_[present_stage_id + 1] += layers[start_id[present_stage_id] + begin_layer_id + i]*straggle[present_stage_id+1]
        for i in range(present_stage_id + 1, 4):
            # change 4 to stage_num_sum
            list_.append(stages_[i])
            communication_list.append(communicaiton[i])
        max_time = compute_pipline_time(list_, communication_list)
        return max_time

    def compute_time_3(begin_layer_id, end_layer_id):
        stages_ = copy.copy(stages)
        stages_[present_stage_id] = 0
        for i in range(end_layer_id - begin_layer_id):
            stages_[present_stage_id] += layers[start_id[present_stage_id] + begin_layer_id + i]*straggle[present_stage_id]
        return stages_[present_stage_id]
    for i in range(num_layer[present_stage_id]+1):
        for j in range(num_layer[present_stage_id]+1):
            if i>=j:
                record[i][j]=float('inf')
            else:
                if present_stage_id-1>=0:
                    time1=compute_time_1(0,i)#pre part
                else:
                    if i!=0:
                        record[i][j] = float('inf')
                        continue
                    time1=0
                time2=compute_time_3(i,j)#medium part f+b
                if present_stage_id+1<=3:
                    #change 3 to last stage id
                    time3=compute_time_2(j,num_layer[present_stage_id])#last part
                else:
                    if j!=num_layer[present_stage_id]:
                        record[i][j]=float('inf')
                        continue
                    time3=0
                if time1==0:
                    record[i][j]=compute_pipline_time([time2,time3],[communicaiton[j-1]])
                elif time3==0:
                    record[i][j]=compute_pipline_time([time1,time2],[communicaiton[start_id[present_stage_id]+i-1]])
                else:
                    tmp=compute_pipline_time([time1,time2],[communicaiton[start_id[present_stage_id]+i-1]])
                    record[i][j]=compute_pipline_time([tmp,time3],[communicaiton[start_id[present_stage_id]+j-1]])
    min_index=find_min_index_2d(record)
    print(record[min_index[0]][min_index[1]])
    return min_index

if __name__ == '__main__':
    main()

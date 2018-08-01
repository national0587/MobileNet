#!/usr/bin/env python
# -*- Coding: utf-8 -*-
from mobilenet import MobileNet
from chainer import training
from chainer.training import extensions
import numpy as np
import chainer
import argparse
import random
import chainer.links as L
from chainer.datasets import split_dataset_random
# If you do not use PlotReport , you can comment out following lines...
import matplotlib
matplotlib.use('Agg')
import chainermn
print(chainer.__version__)


'''
ChainerMN対応版
'''


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype(np.float32)
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


def main():
    parser = argparse.ArgumentParser(
        description='Learning MobileNet')
    parser.add_argument('train', help='Path to training image-label list file')
    # parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--batchsize', '-B', type=int, default=50,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--communicator', type=str,
                        default='hierarchical', help='Type of communicator')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    if args.gpu:
        if args.communicator == 'naive':
            print("Error: 'naive' communicator does not support GPU.\n")
            exit(-1)
        comm = chainermn.create_communicator(args.communicator)
        device = comm.intra_rank
    else:
        if args.communicator != 'naive':
            print('Warning: using naive communicator '
                  'because only naive supports CPU-only execution')
        comm = chainermn.create_communicator('naive')
        device = -1

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        if args.gpu:
            print('Using GPUs')
        print('Using {} communicator'.format(args.communicator))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    model = MobileNet()

    model = L.Classifier(model)

    # model = L.Classifier(MobileNet())
    if args.initmodel:
        print('Load model from {}'.format(args.initmodel))
        chainer.serializers.load_npz(args.initmodel, 224)
    if device >= 0:
        chainer.backends.cuda.get_device_from_id(device).use()  # Make the GPU current
        model.to_gpu()

    # Create a multi node optimizer from a standard Chainer optimizer.
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9), comm)
    optimizer.setup(model)

    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    if comm.rank == 0:
        mean = np.load(args.mean)
        train_val = PreprocessedDataset(args.train, args.root, mean,224)
        train, valid = split_dataset_random(train_val, int(len(train_val)*0.9), seed=0)
    else:
        train, valid = None, None

    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    valid = chainermn.scatter_dataset(valid, comm, shuffle=True)

    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    # val_iter = chainer.iterators.MultiprocessIterator(
    #     valid, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (1 if args.test else 1000), 'iteration'
    log_interval = (1 if args.test else 1000), 'iteration'
    if comm.rank == 0:
        # print("val_interval:{}".format(val_interval))

        # trainer.extend(extensions.Evaluator(val_iter, model, device=device),
        #               trigger=val_interval)
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.snapshot(), trigger=val_interval)
        trainer.extend(extensions.snapshot_object(
            model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
        # Be careful to pass the interval directly to LogReport
        # (it determines when to emit log rather than when to read observations)

        # Save two plot images to the result dir
        if extensions.PlotReport.available():
            print("Plot available")
            trainer.extend(
                extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                      'iteration', file_name='loss.png'))
            trainer.extend(
                extensions.PlotReport(
                    ['main/accuracy', 'validation/main/accuracy'],
                    'iteration', file_name='accuracy.png'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy', 'lr'
        ]))

        trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()

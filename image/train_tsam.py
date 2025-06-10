import argparse
import torch
import wandb

import torch.nn as nn
from torchvision.models import *

from model.wide_res_net import WideResNet
from model.resnet import ResNet18
from model.nfresnet import nf_resnet18
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from data.cifar100 import Cifar100
from data.cifar_224 import Cifar_224
from data.dtd import DTD
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import sys; sys.path.append("..")
from sam import SAM
from tsam import TSAM



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--sampling", default='random1', type=str, help="Choose from random heuristic or hmc.")
    parser.add_argument("--num_samples", default=3, type=int, help="Number of sampled perturbations.")
    parser.add_argument("--tilt", default=1, type=float, help="Tilting hyperparameter.")
    parser.add_argument("--radius", default=1, type=float, help="Radius of the Lp ball of random noise.")
    parser.add_argument("--alpha", default=0.01, type=float, help="Learning rate in HMC.")
    parser.add_argument("--sigma", default=1, type=float, help="Gaussian std in momentum in HMC.")
    parser.add_argument("--HMC_iters", default=1, type=int, help="How many iters to generate one epsilon in HMC.")
    parser.add_argument("--model", default='wideresnet', type=str, help="What model architecture to use.")
    parser.add_argument("--dataset", default='cifar10', type=str, help="Which dataset to evaluate on.")
    parser.add_argument("--corruption", default=0, type=float, help="Ratio of label noise on training data.")
    parser.add_argument("--schedule_t", default=0, type=int, help="0: not schedule, 1: from 0 to tilt, 2: from tilt to 0.")
    parser.add_argument("--eps", default=0.05, type=float, help="Gaussian perturbation when evaluating smoothness.")
    parser.add_argument("--scaling", default=1, type=float, help="Constant for sampling.")
    parser.add_argument("--save", default=False, type=bool, help="True if save the trained models.")
    parser.add_argument("--epsilon", default='gaussian', type=str, help="How do we sample epsilon: gaussian or uniform.")
    args = parser.parse_args()
    print(args)
    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    if args.dataset == 'cifar10':
        dataset = Cifar(args.batch_size, args.threads, args.corruption)
        num_classes = 10
    elif args.dataset == 'cifar100':
        dataset = Cifar100(args.batch_size, args.threads)
        num_classes = 100
    log = Log(log_each=10)
    if args.model == 'wideresnet':
        model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=num_classes)
    elif args.model == 'nfresnet':
        model = nf_resnet18()
    elif args.model == 'resnet':
        model = ResNet18()
    elif args.model == 'vit':
        model = get_model("vit_b_16", weights="DEFAULT")
        num_features = model.heads.head.in_features
        if args.dataset == 'cifar10':
            num_classes = 10
            dataset = Cifar_224(args.batch_size, args.threads)
        elif args.dataset == 'dtd':
            num_classes = 47
            dataset = DTD(args.batch_size, args.threads)
        model.heads.head = nn.Linear(num_features, num_classes)
    else:
        model = None
    model = model.to(device)
    base_optimizer = torch.optim.SGD
    optimizer = TSAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    wandb.init(
        # set the wandb project where this run will be logged
        project="sam",
        group="tsam"
        
        config={
        "model": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "bs": args.batch_size,
        "lr": args.learning_rate,
        "moment parameter": args.momentum,
        "weight_decay": args.weight_decay,
        "rho": args.rho,
        "sampling": args.sampling,
        "num_samples": args.num_samples,
        "tilt": args.tilt,
        "radius": args.radius,
        "corruption": args.corruption,
        "eps": args.eps,
        "schedule_t": args.schedule_t,
        "scaling": args.scaling,
        "epsilon": args.epsilon,
        }
    )

    total_step = args.epochs * len(dataset.train)
    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for inner_iter, batch in enumerate(dataset.train):
            inputs, targets = (b.to(device) for b in batch)
            grads = []
            losses = []
            if args.sampling == 'random0': # completely uniformly at random
                for i in range(args.num_samples):
                    grad = dict()
                    enable_running_stats(model)
                    optimizer.first_step2(z=args.radius, scaling=args.scaling, zero_grad=True) # theta <- theta + Gaussian
                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                    loss.mean().backward()
                    optimizer.set_old_theta(zero_grad=False)
                    for p_name, p in model.named_parameters():
                        grad[p_name] = p.grad.clone()
                    losses.append(loss.mean().item())
                    grads.append(grad)
                    optimizer.zero_grad()

            elif args.sampling == 'random1': # ours
                for i in range(args.num_samples):
                    grad = dict()

                    enable_running_stats(model)
                    optimizer.first_step2(epsilon=args.epsilon, r=args.radius, scaling=args.scaling, zero_grad=True)
                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                    loss.mean().backward()
                    optimizer.first_step3(zero_grad=True)

                    disable_running_stats(model)
                    loss2 = smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing)
                    loss2.mean().backward()
                    optimizer.set_old_theta(zero_grad=False)

                    for p_name, p in model.named_parameters():
                        grad[p_name] = p.grad.clone()

                    losses.append(loss2.mean().item())
                    grads.append(grad)
                    optimizer.zero_grad()

            t = args.tilt
            if args.schedule_t == 1:
                t = (epoch * len(dataset.train) + inner_iter) / total_step * args.tilt
            elif args.schedule_t == 2:
                t = (total_step - epoch * len(dataset.train) - inner_iter) / total_step * args.tilt

            optimizer.tilted_aggregation(model, losses, grads, t)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

        #if args.save:
        #    if epoch % 20 == 0 or epoch == 199:
        #        torch.save(model.state_dict(), "./checkpoints/tsam/"+str(epoch)+"_rho"+str(args.rho)+"_lr"+str(args.learning_rate)+".pth")




    






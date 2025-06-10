import argparse
import torch
import wandb

import torch.nn as nn
from torchvision.models import *
import numpy as np


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
    parser.add_argument("--model", default='wideresnet', type=str, help="What model architecture to use.")
    parser.add_argument("--dataset", default='cifar10', type=str, help="Which dataset to evaluate on.")
    parser.add_argument("--corruption", default=0, type=float, help="Ratio of label noise on training data.")
    parser.add_argument("--eps", default=0.05, type=float, help="Gaussian perturbation when evaluating smoothness.")
    

    parser.add_argument("--sampling", default='random1', type=str, help="Choose from random heuristic or hmc.")
    parser.add_argument("--num_samples", default=3, type=int, help="Number of sampled perturbations.")
    parser.add_argument("--tilt", default=1, type=float, help="Tilting hyperparameter.")
    parser.add_argument("--radius", default=1, type=float, help="Radius of the Lp ball of random noise.")
    parser.add_argument("--alpha", default=0.01, type=float, help="Learning rate in HMC.")
    parser.add_argument("--sigma", default=1, type=float, help="Gaussian std in momentum in HMC.")
    parser.add_argument("--HMC_iters", default=1, type=int, help="How many iters to generate one epsilon in HMC.")
    parser.add_argument("--schedule_t", default=0, type=int, help="0: not schedule, 1: from 0 to tilt, 2: from tilt to 0.")
    parser.add_argument("--scaling", default=1, type=float, help="Constant for sampling.")
    parser.add_argument("--method", default='sgd', type=str, help="which method model to eval")
    

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
        model = get_model("vit_b_16", weights=None)
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
	sam_optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	wandb.init(
    # set the wandb project where this run will be logged
    project="sam",
    group=args.method+"-useless",

    config={
    'model': args.model,
    'dataset': args.dataset,
    'clipped noise': False
    }
	)


	models = {"sgd": "sgd/lr0.1", "sam": "sam/rho0.1_lr0.003", "tsam": "tsam/epoch199_rho0.1_lr0.003_num_samples3_tilt20.0_scaling1"}
	#checkpoint = torch.load("./checkpoints/tsam/epoch199_rho0.1_lr0.003_num_samples3_tilt20.0_scaling1")
	checkpoint = torch.load("./checkpoints/"+models[args.method])
	model.load_state_dict(checkpoint)

	def get_accuracy_and_loss(data):
    	loss = 0
    	num_samples = 0
    	tot_correct = 0

    	with torch.no_grad():
        	for batch in data:
            	inputs, targets = (b.to(device) for b in batch)
            	predictions = model(inputs)
            	loss_flatten = smooth_crossentropy(predictions, targets)
            	correct = torch.argmax(predictions, 1) == targets
            	loss += loss_flatten.sum().item()
            	num_samples += loss_flatten.size(0)
            	tot_correct += correct.sum().item()
            # print(num_samples)
    	return loss, num_samples, tot_correct / num_samples

	def get_sam_loss(data):
    	sam_optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    	for i, batch in enumerate(data):
        	inputs, targets = (b.to(device) for b in batch)
        	predictions = model(inputs)
        	loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
        	loss.mean().backward()
        	sam_optimizer.first_step(zero_grad=True)
        	break # only run one mini-batch
    	train_loss, _ = get_accuracy_and_loss(dataset.train)
    	test_loss, _ = get_accuracy_and_loss(dataset.test)
    	return train_loss, test_loss

	def get_tsam_loss(data):
    	tsam_optimizer = TSAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    	train_losses = []
    	test_losses = []
    	for i, batch in enumerate(data):
        	for num_sample in range(args.num_samples):
            	inputs, targets = (b.to(device) for b in batch)
            	tsam_optimizer.first_step2(r=args.radius, scaling=1, zero_grad=True) # theta' <- theta + Gaussian
            	predictions = model(inputs)
            	loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            	loss.mean().backward()
            	tsam_optimizer.first_step3(zero_grad=False) # theta' <- theta' + alpha * L'(theta')
            	train_loss, _ = get_accuracy_and_loss(dataset.train)
            	test_loss, _ = get_accuracy_and_loss(dataset.test)
            	train_losses.append(train_loss)
            	test_losses.append(test_loss)
                tsam_optimizer.set_old_theta(zero_grad=True)
            break
        train_loss = 1.0 / args.tilt * np.log(1.0 / args.num_samples * np.sum(np.exp(args.tilt * np.asarray(train_losses))))
        test_loss = 1.0 / args.tilt * np.log(1.0 / args.num_samples * np.sum(np.exp(args.tilt * np.asarray(test_losses))))
    return train_loss, test_loss

	# model.eval()
	train_loss, train_accu = get_accuracy_and_loss(dataset.train)
	test_loss, test_accu = get_accuracy_and_loss(dataset.test)
	# train_loss, test_loss = get_tsam_loss(dataset.test)
	print("First do sanity check on the loaded model...")
	print("train loss:", train_loss)
	print("train accu:", train_accu)
	print("test loss:", test_loss)
	print("test accu:", test_accu)

	old_p = dict()
	for p_name, p in model.named_parameters():
    	old_p[p_name] = p.data.clone() # save the solutions before perturbation

	losses = []
	noise_norm = 0
	for num_samples in range(500):
    	for p_name, p in model.named_parameters():

        	tmp = args.eps * torch.randn_like(p.data)
        	p.data = old_p[p_name] + tmp
        	noise_norm += torch.square(torch.norm(tmp))
        noise_norm = torch.sqrt(noise_norm)
    	train_loss, train_accu = get_accuracy_and_loss(dataset.train)
    	losses.append(train_loss)
    	print(noise_norm.item(), 'loss:', train_loss)

	avg_loss = sum(losses) / len(losses)
	var_loss = sum([(item - avg_loss) ** 2 for item in losses]) / len(losses)
	wandb.log({"num_samples": 500, "eps": args.eps, "average perturbed loss": avg_loss, "var perturbed loss": var_loss})

	print(sum(losses) / len(losses))

















    
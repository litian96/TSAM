import torch
import tqdm
import argparse
import random
import numpy as np
import wandb
import evaluate

import datasets
from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import matthews_corrcoef

from get_data import get_data

import sys; sys.path.append("..")
from sam import SAM
from tsam import TSAM


def initialize(args, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


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
    parser.add_argument("--sampling", default='random2', type=str, help="Choose from random heuristic or hmc.")
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

    train_data, test_data = get_data(args.dataset, DistilBertTokenizer.from_pretrained('distilbert-base-uncased'))

    num_classes = {"cola": 2, "wnli": 2, "sst2": 2, "stsb": 1, 
                   "mnli": 3, "mnli_matched": 3, "mnli_mismatched": 3, 
                   "ax": 2, "mrpc": 2, "qnli": 2, "qqp": 2, "rte": 2}

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",  # Use the DistilBERT model, with an uncased vocab.
        num_labels=num_classes[args.dataset],  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    model = model.to(device)
    base_optimizer = torch.optim.SGD
    optimizer = TSAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, 
                    momentum=args.momentum, weight_decay=args.weight_decay)

    pearson_corr_eval = evaluate.load("pearsonr")
    mse_eval = evaluate.load('mse')
    spearman_corr_eval = evaluate.load('spearmanr')
    mse_loss = torch.nn.MSELoss()
    
    wandb.init(
            # set the wandb project where this run will be logged
            project="sam",
            group="tsam",

            # track hyperparameters and run metadata
            config={
            "model": "distilbert",
            "dataset": args.dataset,
            "epochs": args.epochs,
            "bs": args.batch_size,
            "lr": args.learning_rate,
            "moment parameter": args.momentum,
            "weight_decay": args.weight_decay,
            "eps": args.eps,
            "rho": args.rho,
            "corruption": args.corruption,
            "sampling": args.sampling,
            "num_samples": args.num_samples,
            "tilt": args.tilt,
            "radius": args.radius,
            "schedule_t": args.schedule_t,
            "scaling": args.scaling,
            "epsilon": args.epsilon,
            }
    )

    for epoch_i in tqdm.trange(0, args.epochs):
        train_loss = 0
        train_accuracy = 0
        count = 0

        model.train()
        for step, batch in enumerate(train_data.shuffle(epoch_i).iter(args.batch_size)):
            b_input_ids = torch.tensor(batch['input_ids'])[:,0].to(device)
            b_input_mask = torch.tensor(batch['attention_mask'])[:,0].to(device)
            b_labels = torch.tensor(batch['label']).to(device)

            model.zero_grad()
            result = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
            
            logits = result.logits.squeeze()
            loss = mse_loss(logits, b_labels)

            train_loss += loss.item() * b_input_ids.shape[0]
            count += b_input_ids.shape[0]

            grads = []
            losses = []

            for i in range(args.num_samples):

                grad = dict()
                
                optimizer.first_step2(epsilon=args.epsilon, r=args.radius, scaling=args.scaling, zero_grad=True)
                result = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
                logits = result.logits.squeeze()
                loss = mse_loss(logits, b_labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.first_step3(zero_grad=True)

                result = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
                logits = result.logits.squeeze()
                loss2 = mse_loss(logits, b_labels)

                loss2.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.set_old_theta(zero_grad=False)

                for p_name, p in model.named_parameters():
                    grad[p_name] = p.data.clone()

                losses.append(loss2.item())
                grads.append(grad)
                optimzier.zero_grad()

            t = args.tilt
            if args.schedule_t == 1:
                t = (epoch * len(dataset.train) + inner_iter) / total_step * args.tilt
            elif args.schedule_t == 2:
                t = (total_step - epoch * len(dataset.train) - inner_iter) / total_step * args.tilt

            optimizer.tilted_aggregation(model, losses, grads, t)


        train_loss = train_loss / count
        print("training loss:", train_loss)
        wandb.log({"training loss": train_loss, "epoch": epoch_i})

        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for step, batch in enumerate(test_data.iter(args.batch_size)):
                b_input_ids = torch.tensor(batch['input_ids'])[:,0].to(device)
                b_input_mask = torch.tensor(batch['attention_mask'])[:,0].to(device)
                b_labels = torch.tensor(batch['label']).to(device)

                result = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
                logits = result.logits.squeeze()
                

                all_predictions.extend(logits.cpu().numpy())
                all_labels.extend(b_labels.cpu().numpy())

        test_mse = mse_eval.compute(predictions=all_predictions, references=all_labels)['mse']
        test_pearson = pearson_corr_eval.compute(predictions=all_predictions, references=all_labels)['pearsonr']
        test_spearman = spearman_corr_eval.compute(predictions=all_predictions, references=all_labels)['spearmanr']

        wandb.log({"test mse": test_mse, "test pearson": test_pearson, "test spearman": test_spearman, "epoch": epoch_i})




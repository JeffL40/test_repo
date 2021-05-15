import argparse
import math
import sys
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from fast_transformers.masking import LengthMask, TriangularCausalMask
from fast_transformers.builders import TransformerEncoderBuilder

from utils import add_optimizer_arguments, get_optimizer, \
    add_transformer_arguments, print_transformer_arguments, \
    EpochStats, load_model, save_model, loss_fn, train, evaluate, add_auxiliary_arguments,\
    plot_hidden_state_2d, extract_hidden_state

from datasets import CopyTask, CountTask, CountTaskWithEOS

from modules import SequencePredictorRNN, SequencePredictorRecurrentTransformer

from constants import device

def main(argv=None):
    # Choose a device and move everything there
    print("Running on {}".format(device))

    parser = argparse.ArgumentParser(
        description="Train a transformer for a copy task"
    )
    add_optimizer_arguments(parser)
    add_transformer_arguments(parser)
    add_auxiliary_arguments(parser)
    args = parser.parse_args(argv)
    print("args:\n-----\n", args)
    # Make the dataset and the model
    train_set = CountTaskWithEOS(args.sequence_length)
    test_set = CountTaskWithEOS(args.sequence_length)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        pin_memory=device=="cuda"
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        pin_memory=device=="cuda"
    )

    if args.model_type == "transformer":
        model = SequencePredictorRecurrentTransformer(
                    d_model=args.d_model, n_classes=args.n_classes,
                    sequence_length=args.sequence_length,
                    attention_type=args.attention_type,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    d_query=args.d_model, # used to be d_query
                    dropout=args.dropout,
                    softmax_temp=None,
                    attention_dropout=args.attention_dropout,
                )
    else:
        model = SequencePredictorRNN(
                    d_model=args.d_model, n_classes=args.n_classes,
                    n_layers=args.n_layers,
                    dropout=args.dropout,
                    rnn_type=args.model_type
                )
    print(f"Created model:\n{model}")
    model.to(device)
    # Start training
    optimizer = get_optimizer(model.parameters(), args)
    start_epoch = 1
    if args.continue_from:
        start_epoch = load_model(
            args.continue_from,
            model,
            optimizer,
            device
        )
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: 1. if e < args.reduce_lr_at else 0.1
    )
    for e in range(start_epoch, args.epochs+1):
        train(model, optimizer, train_loader, device)
        print('Epoch:', e)
        evaluate(model, test_loader, device)
        if (e % args.save_frequency) == 0 and args.save_to:
            save_model(args.save_to, model, optimizer, e)
        lr_schedule.step()
    if args.plot_hidden:
        x, y, m = test_set.__next__() # x is 1d of length sequence_len
        model.eval()
        yhat = model(x.unsqueeze(1))
        hdn = model.hidden_state # batch x seq x hdn
        max_len = 10
        print("Plotting on: ",x[:max_len])
        plot_hidden_state_2d(hdn[0,:max_len,:].detach().numpy(), pca=True)
if __name__ == "__main__":
    main()
    """
    To train up an RNN, do something like:

    python main.py --n_classes=3 --epochs=20 --model_type=rnn --d_model=3 --save_to=models_storage/model_rnn --plot_hidden=True

    Can change from rnn to `lstm` or `transformer`. 
    Plots hidden state movement on random sequence after training.
    If you don't want to train and just want to plot, set --epochs=0.
    """

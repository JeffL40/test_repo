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

from constants import device, running_on_colab

if running_on_colab:
    extra_paths = ['', '/content', '/env/python', '/usr/lib/python37.zip', '/usr/lib/python3.7', '/content/drive/My Drive/final_project_material', '/usr/lib/python3.7/lib-dynload', '/usr/local/lib/python3.7/dist-packages', '/usr/lib/python3/dist-packages', '/usr/local/lib/python3.7/dist-packages/IPython/extensions', '/root/.ipython']
    for p in extra_paths:
        if p not in sys.path:
            sys.path.append(p)

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
    for model_type in ['rnn', 'lstm', 'transformer']:
        for max_depth in range(1, 12):
            train_set = CountTaskWithEOS(args.sequence_length, max_depth=max_depth)
            test_set = CountTaskWithEOS(args.sequence_length, max_depth=max_depth)
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

            if model_type == "transformer":
                d_model = 8
                model = SequencePredictorRecurrentTransformer(
                            d_model=d_model, n_classes=args.n_classes,
                            sequence_length=args.sequence_length,
                            attention_type=args.attention_type,
                            n_layers=args.n_layers,
                            n_heads=args.n_heads,
                            d_query=d_model, # used to be d_query
                            dropout=args.dropout,
                            softmax_temp=None,
                            attention_dropout=args.attention_dropout,
                        )
            else:
                d_model=3
                model = SequencePredictorRNN(
                            d_model=d_model, n_classes=args.n_classes,
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
                print('Epoch:', e)
                print('Training...')
                train(model, optimizer, train_loader, device)
                print('Evaluating...')
                acc = evaluate(model, test_loader, device, return_accuracy=True)
                if (e % args.save_frequency) == 0 and args.save_to:
                    save_model("model_" + model_type + "_depth_" + str(max_depth), 
                    model, optimizer, e)
                lr_schedule.step()
                if acc >= 1:
                    break
if __name__ == "__main__":
    main()
    """
    Run this:

    python train_many.py --n_classes=3 --epochs=100

    """

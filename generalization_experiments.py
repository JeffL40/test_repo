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
    print("Running on {}".format(device))
    parser = argparse.ArgumentParser(
        description="Train a transformer for a copy task"
    )
    add_optimizer_arguments(parser)
    add_transformer_arguments(parser)
    add_auxiliary_arguments(parser)
    args = parser.parse_args(argv)
    print("args:\n-----\n", args)
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
    model.load_state_dict(torch.load(args.continue_from, map_location=device)['model_state'])

    # Make the dataset and the model
    test_set = CountTaskWithEOS(args.sequence_length)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        pin_memory=device=="cuda"
    )
    if args.plot_hidden:
        x, y, m = test_set.__next__() # x is 1d of length sequence_len
        model.eval()
        yhat = model(x.unsqueeze(1))
        hdn = model.hidden_state # batch x seq x hdn
        max_len = 10
        print("Plotting on: ",x[:max_len])
        plot_hidden_state_2d(hdn[0,:max_len,:].detach().numpy(), pca=True)

    """
    > models_storage
    >> rnn
    >>> model_0
    >>> model_1
    >> lstm
    >>> model_0
    >> transformer
    >>> model_0

    Do a few things to test generalization.
    For a given model `model`:
        Generate some sequences (mix of easy and hard). For a sequence `x`,
            1. Run `model` on `x`.
            2. Check whether `model` got the answer completely right.
            3. Plot the hidden state space mov't wrto. `x` to interpret
            `model` solution and why it got the answer wrong/right
    """
if __name__ == "__main__":
    main()

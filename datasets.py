import torch
from torch.utils.data import DataLoader, IterableDataset
import random
import numpy as np

class CopyTask(IterableDataset):
    def __init__(self, max_sequence, n_classes):
        self._max_sequence = max_sequence
        self._n_classes = n_classes
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Make some local copies
        max_seq = self._max_sequence
        n_classes = self._n_classes

        # Generate the random sequence
        n = torch.randint(max_seq//4, (max_seq-1)//2, tuple())
        random_sequence = (torch.rand(n)*n_classes).long() + 1

        # Generate the input, target and loss mask
        x = torch.zeros(max_seq, dtype=torch.long)
        y = torch.zeros(max_seq, dtype=torch.long)
        mask = torch.zeros(max_seq)
        x[:n] = random_sequence
        x[n+1:2*n+1] = random_sequence
        y[:-1] = x[1:]
        mask[n-1:2*n] = 1
        return x, y, mask

class CountTask(IterableDataset):
    def __init__(self, max_sequence):
        self.max_sequence = max_sequence

    def __iter__(self):
        return self

    def __next__(self):
        tok_a = 0
        tok_b = 1
        # Make some local copies
        lengths = random.choices(range(1,12), weights=(10, 6, 4, 3, 1, 1, 1, 1, 1, 1, 1), k=10)
        x = []
        for length in lengths:
            if len(x) + length*2 <= self.max_sequence:
                x += [tok_a]*length + [tok_b]*length
            else:
                x += [tok_a]*(self.max_sequence - len(x))
                break
        x += [tok_a]*(self.max_sequence - len(x))
        y = x[1:] + [tok_a]
        mask = [1 if x[i] == tok_b else 0 for i in range(len(x))]
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.float64)
        return x, y, mask

class CountTaskWithEOS(IterableDataset):
    def __init__(self, max_sequence, max_depth=12):
        self.max_sequence = max_sequence
        self.max_depth = max_depth

    def __iter__(self):
        return self

    def __next__(self):
        tok_EOS = 0
        tok_a = 1
        tok_b = 2
        
        
        """
        Replace with exponential distribution?
        """
        lengths = random.choices(range(1,12), weights=(10, 6, 4, 3, 1, 1, 1, 1, 1, 1, 1), k=10)



        x = []
        for length in lengths:
            if len(x) + length * 2 + 1 <= self.max_sequence:
                for _ in range(length):
                    x.append(tok_a)
                for _ in range(length):
                    x.append(tok_b)
                x.append(tok_EOS)
            else:
                break
        for _ in range(self.max_sequence - len(x)):
            x.append(tok_EOS)
        y = x[1:] + [tok_EOS]
        mask = [1 if x[i] == tok_b else 0 for i in range(len(x))]
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.float64)
        return x, y, mask


if __name__ == "__main__":
    # gen = np.random.default_rng()
    # print(gen)
    # for _, x in zip(range(10), gen):
    #     print(x)
    # print(np.floor(abs(gen.normal(1, 6, 6)-1)+1))
    c = CountTaskWithEOS(128)
    c.__next__()
    # gen = normal_int_gen(5)
    # print(next(gen))
    pass

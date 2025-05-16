#!/usr/bin/env python

import torch


class BernoulliDatastream(torch.utils.data.IterableDataset):
    class BernoulliIterator():
        def __init__(self, p, N):
            self.p = p
            self.N = N
            self.Bernoulli = torch.distributions.bernoulli.Bernoulli(p)
            self.n = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.n >= self.N:
                raise StopIteration
            self.n += 1

            b = torch.tensor([self.Bernoulli.sample()])
            return b, b
                

    def __init__(self, p, N):
        super(BernoulliDatastream).__init__()
        self.p = p
        self.N = N       # virtual size of the dataset (defines epoch)

    def __len__(self):
        return self.N

    def __iter__(self):
        return BernoulliDatastream.BernoulliIterator(self.p, self.N)


if __name__ == '__main__':
    import sys
    p = float(sys.argv[1]) if len(sys.argv) > 1 else  0.5
    N =   int(sys.argv[2]) if len(sys.argv) > 2 else 10

    import numpy as np
    bsum = 0
    berns = BernoulliDatastream(p, N)
    for n, (b, vb) in enumerate(berns):
        bsum += b
    print(bsum / N)

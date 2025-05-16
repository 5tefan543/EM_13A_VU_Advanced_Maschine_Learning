#!/usr/bin/env python

import torch
import torchvision as tv
import numpy as np


class BarDatastream(torch.utils.data.IterableDataset):
    class BarIterator():
        def __init__(self, sqsize, oris, N, random):
            self.sqsize = sqsize # image width and height in pixels
            self.oris = oris     # 2 or 4
            self.random = random
            self.rng = np.random.default_rng() if self.random else None
            winfo = torch.utils.data.get_worker_info()
            self.N = int(np.ceil(N / winfo.num_workers)) if winfo else N
            self.n = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.n >= self.N:
                raise StopIteration
            self.n += 1

            ori = self.rng.integers(0, self.oris) if self.random \
                else self.n % self.oris
            if ori < 2:
                img = torch.zeros((self.sqsize, self.sqsize))
                if ori:
                    img[self.sqsize//2, :] = 1
                else:
                    img[:, self.sqsize//2] = 1
            else:
                img = torch.eye(self.sqsize)
                if ori == 2:
                    img = img.flipud()

            return img, torch.tensor(ori)
                

    def __init__(self, sqsize, oris, N, random=False):
        super(BarDatastream).__init__()
        self.sqsize = sqsize
        self.oris = oris
        self.N = N              # virtual size of the dataset (defines epoch)
        self.random = random

    def __len__(self):
        return self.N

    def __iter__(self):
        return BarDatastream.BarIterator(self.sqsize, self.oris, self.N,
                                         self.random)


if __name__ == '__main__':
    import sys
    sqsize = int(sys.argv[1]) if len(sys.argv) > 1 else  3
    oris   = int(sys.argv[2]) if len(sys.argv) > 2 else  4
    N      = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    savebase =   sys.argv[4]  if len(sys.argv) > 4 else None

    plines = BarDatastream(sqsize, oris, N, False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    for img, cls in plines:
        ax.clear()
        ax.imshow(tv.transforms.ToPILImage()(img), vmin=0, vmax=255)
        ax.set_axis_off()
        if savebase:
            fig.savefig(f'{savebase}-{cls}.png',
                        bbox_inches='tight', pad_inches=0)
        else:
            plt.pause(0.1)
    plt.close(fig)

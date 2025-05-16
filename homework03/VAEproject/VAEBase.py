import torch
from torch import nn
import torchvision as tv
import matplotlib.pyplot as plt


class VAEBase(nn.Module):
    def save(self, filebase):
        torch.save(self.state_dict(), filebase + '.pth')

    def load(self, filebase):
        self.load_state_dict(torch.load(filebase + '.pth'))
        
    def sample(self, z): # z is either tensor (latent vector) or int (zdim)
        # assumes that generated image is square
        self.eval()
        with torch.no_grad():
            if not isinstance(z, torch.Tensor):
                z = self.distrib_pz.sample(torch.Size([z]))
            dec = self.dec(z)
            sqsize = int(torch.sqrt(torch.tensor(dec.shape[0])))
            return dec.reshape(torch.Size([sqsize, sqsize]))

    def plotDecRandom(self, zdim, M=8, N=8):
        fig, ax = plt.subplots(M, N)
        for m in range(M):
            for n in range(N):
                ax[m][n].imshow(tv.transforms.ToPILImage()(self.sample(zdim)),
                                vmin=0, vmax=255)
                ax[m][n].set_axis_off()
        fig.suptitle(f"dec($z$) with $z \sim p(z)$")
        plt.show()
        plt.close(fig)

    def plotDecGrid(self, zdim, N=9, savename=None):
        if zdim < 1 or zdim > 2:
            print(f"Cannot visualize {zdim}D z space")
            return
        mumax = 2
        M = 1 if zdim == 1 else N
        fig, ax = plt.subplots(M, N, squeeze=False,
                               figsize=(6.4, 1.2 if zdim < 2 else 4.8))
        for m in range(M):
            m_ = M - 1 - m
            for n in range(N):
                zn = -mumax + n  / (N-1) * 2 * mumax
                z = torch.tensor([zn]) if zdim == 1 else \
                    torch.tensor([-mumax + m_  / (M-1) * 2 * mumax, zn])
                ax[m][n].imshow(tv.transforms.ToPILImage()(self.sample(z)),
                                vmin=0, vmax=255)
                # hide axes except at bottom and left, which annotate:
                ax[m][n].axes.set_frame_on(False)
                ax[m][n].axes.get_xaxis().set_visible(False)
                ax[m][n].axes.get_yaxis().set_visible(False)
                ax[m][n].axes.tick_params(labelleft=False, labelbottom=False)
                ax[m][n].axes.set_xticks([])
                ax[m][n].axes.set_yticks([])
                if n == 0 and zdim > 1:
                    ax[m][n].axes.get_yaxis().set_visible(True)
                    ax[m][n].set_ylabel(f'{z[0].item():.2f}')
                if m == M - 1:
                    ax[m][n].axes.get_xaxis().set_visible(True)
                    ax[m][n].set_xlabel(f'{z[0 if zdim == 1 else 1].item():.2f}')
        fig.suptitle(f"dec($z$) with $z$ regularly sampled from {zdim}D grid")
        if savename:
            fig.savefig(savename, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.close(fig)

import torch
import torchvision as tv
import matplotlib.pyplot as plt


class AEML:
    def __init__(self, dataloader, model, loss_fn, optimizer):
        self.dataloader = dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.fig, self.ax = None, None

    def train(self):
        self.model.train()
        for n_batch, (imgs_in, imgs_out) in enumerate(self.dataloader):
            pred = self.model(imgs_in)
            loss = self.loss_fn(pred,
                                imgs_out if pred.shape == imgs_out.shape
                                else imgs_in.reshape(pred.shape))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for imgs_in, imgs_out in self.dataloader:
                pred = self.model(imgs_in)
                test_loss += \
                    self.loss_fn(pred,
                                 imgs_out if pred.shape == imgs_out.shape
                                 else imgs_in.reshape(pred.shape)).item()
        if self.fig:
            self.showInstances(imgs_in[0], pred[0])
        return test_loss / len(self.dataloader)

    def run(self, N_epochs, savename=None):
        for epoch in range(N_epochs):
            if savename:
                self.model.save(f'{savename}-{epoch:03d}')
            final_train_loss = self.train()
            test_loss = self.test()
            print(f"Epoch{epoch:4d} {final_train_loss=:.3f} {test_loss=:.3f}")
        if savename:
            self.model.save(f'{savename}-{epoch+1:03d}')

    # Only the following methods are specific to AE; the above is generic NN:

    def openFigure(self):
        self.fig, self.ax = plt.subplots(1, 2)

    def showInstances(self, input, output):
        self.ax[0].clear()
        self.ax[0].imshow(tv.transforms.ToPILImage()(input), vmin=0, vmax=255)
        self.ax[0].set_axis_off()
        self.ax[1].clear()
        self.ax[1].imshow(tv.transforms.ToPILImage()
                          (output.reshape(input.shape)), vmin=0, vmax=255)
        self.ax[1].set_axis_off()
        plt.pause(0.1)

    def closeFigure(self):
        plt.close(self.fig)
        self.fig, self.ax = None, None

    def plotEncDataset(self, savename=None):
        self.model.eval()
        zs = None
        classes = None
        with torch.no_grad():
            for img, cls in self.dataloader:
                z = self.model.encode_mu(img)
                if isinstance(zs, torch.Tensor):
                    zs = torch.vstack((zs, z))
                    classes = torch.cat((classes, cls))
                else:
                    zs = z
                    classes = cls
        fig, ax = plt.subplots(figsize=(6.4, 1.8 if zs.shape[1] < 2 else 4.8))
        lim = 5
        if zs.shape[1] < 2:
            zs = torch.hstack((zs, torch.zeros_like(zs)))
            loc = 'upper right'
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.set_box_aspect(0.2)
            ax.axis(xmin=-lim, xmax=lim)
        else:
            loc = 'upper left'
            ax.axis('square')
            ax.axis(xmin=-lim, xmax=lim, ymin=-lim, ymax=lim)
        scatter = ax.scatter(zs[:,0].numpy(), zs[:,1].numpy(),
                             s=max(1000 / len(zs[:,0].unique()), 1),
                             c=classes.numpy())
        ax.add_artist(ax.legend(*scatter.legend_elements(), title='Classes',
                                loc=loc, bbox_to_anchor=(1,1)))
        fig.suptitle(f"Instance means encoded to {z.shape[1]}D z space")
        if savename:
            fig.savefig(savename, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.close(fig)

    def plotReconDataset(self, N = 8):
        self.model.eval()
        with torch.no_grad():
            fig, ax = plt.subplots(2, N, figsize=(6.4, 1.8))
            data = iter(self.dataloader)
            n = 0
            while n < N:
                imgs_in, cls = next(data)         # get batch
                if len(imgs_in.shape) < 3:
                    imgs_in.unsqueeze_(-1)
                imgs_out = self.model.forward(imgs_in).reshape(imgs_in.shape)
                for i in range(imgs_in.shape[0]): # traverse batch
                    ax[0][n].imshow(tv.transforms.ToPILImage()(imgs_in[i]),
                                    vmin=0, vmax=255)
                    ax[1][n].imshow(tv.transforms.ToPILImage()(imgs_out[i]),
                                    vmin=0, vmax=255)
                    ax[0][n].set_axis_off()
                    ax[1][n].set_axis_off()
                    n += 1
                    if n == N:
                        break
            fig.suptitle(f"{N} input images and their reconstructions")
            plt.show()
            plt.close(fig)

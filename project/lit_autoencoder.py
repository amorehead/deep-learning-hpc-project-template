from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST


class LitAutoEncoder(pl.LightningModule):

    def __init__(self, lr: float = 1e-3, save_dir: str = ''):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_mse_loss', loss)
        return loss

    # ---------------------
    # training setup
    # ---------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, self.hparams.num_epochs, eta_min=1e-4)
        metric_to_track = 'train_mse_loss'
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': scheduler,
            'monitor': metric_to_track
        }

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="train_mse_loss", mode="min")
        checkpoint = ModelCheckpoint(monitor="train_mse_loss", save_top_k=3,
                                     dirpath=self.hparams.save_dir,
                                     filename='LitAutoEncoder-{epoch:02d}-{train_mse_loss:.2f}')
        return [early_stop, checkpoint]


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_dataloader_workers', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default="models", help="Directory in which to save models")

    parser.add_argument('--early_stop_callback', type=bool, default=True)
    parser.add_argument('--comet_workspace_name', type=str, default=None)
    parser.add_argument('--comet_project_name', type=str, default=None)
    parser.add_argument('--comet_model_name', type=str, default=None)
    parser.add_argument('--comet_registry_model_name', type=str, default=None)
    parser.add_argument('--comet_registry_model_version', type=str, default='1.0.0')

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Define HPC-specific properties in-file
    args.accelerator = 'ddp'
    args.gpus = 6

    # ------------
    # data
    # ------------
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, num_workers=args.num_dataloader_workers)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size, num_workers=args.num_dataloader_workers)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, num_workers=args.num_dataloader_workers)

    # ------------
    # model
    # ------------
    model = LitAutoEncoder(args.lr, args.save_dir)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.min_epochs = args.num_epochs

    # The arguments made to CometLogger are passed on to the comet_ml.Experiment class
    comet_logger = CometLogger(
        project_name=args.comet_project_name,  # Optional
        experiment_name=f'EarlyStopping-Adam-{args.batch_size}-{args.learning_rate}'  # Optional
    )
    trainer.logger = comet_logger

    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()

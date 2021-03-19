from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST


class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x


class LitClassifier(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-3, save_dir: str = ''):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_cross_entropy', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_cross_entropy', loss, on_step=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_cross_entropy', loss, on_step=True, on_epoch=True, sync_dist=True)

    # ---------------------
    # training setup
    # ---------------------
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, self.hparams.num_epochs, eta_min=1e-4)
        metric_to_track = 'valid_cross_entropy'
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': scheduler,
            'monitor': metric_to_track
        }

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="valid_cross_entropy", mode="min")
        checkpoint = ModelCheckpoint(monitor="valid_cross_entropy", save_top_k=3,
                                     dirpath=self.hparams.save_dir,
                                     filename='LitClassifier-{epoch:02d}-{valid_cross_entropy:.2f}')
        return [early_stop, checkpoint]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


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
    parser.add_argument('--save_dir', type=str, default="models", help="Directory in which to save models")

    parser.add_argument('--early_stop_callback', type=bool, default=True)
    parser.add_argument('--num_dataloader_workers', type=int, default=2)
    parser.add_argument('--comet_workspace_name', type=str, default=None)
    parser.add_argument('--comet_project_name', type=str, default=None)
    parser.add_argument('--comet_model_name', type=str, default=None)
    parser.add_argument('--comet_registry_model_name', type=str, default=None)
    parser.add_argument('--comet_registry_model_version', type=str, default='1.0.0')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
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
    model = LitClassifier(Backbone(hidden_dim=args.hidden_dim), args.lr, args.save_dir)

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
    trainer.early_stop_callback = args.early_stop_callback
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()

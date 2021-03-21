import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
    def __init__(self, backbone, num_epochs: int = 5, lr=1e-4):
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, self.hparams.num_epochs, eta_min=1e-4)
        metric_to_track = 'valid_cross_entropy'
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': metric_to_track
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    parser.add_argument('--multi_gpu_backend', type=str, default='ddp', help="Backend to use for multi-GPU training")
    parser.add_argument('--num_gpus', type=int, default=-1, help="Number of GPUs to use (e.g. -1 = all available GPUs)")
    parser.add_argument('--profiler_method', type=str, default='simple', help="PyTorch Lightning profiler to use")
    parser.add_argument('--num_epochs', type=int, default=5, help="Maximum number of epochs to run for training")
    parser.add_argument('--batch_size', default=4096, type=int, help='Number of samples included in each data batch')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden units in each hidden layer')
    parser.add_argument('--num_dataloader_workers', type=int, default=6, help='Number of CPU threads for loading data')
    parser.add_argument('--experiment_name', type=str, default=None, help="Neptune experiment name")
    parser.add_argument('--project_name', type=str, default='amorehead/DLHPT', help="Neptune project name")
    parser.add_argument('--ckpt_dir', type=str, default="checkpoints", help="Directory in which to save checkpoints")
    parser.add_argument('--ckpt_name', type=str, default=None, help="Filename of best checkpoint")
    args = parser.parse_args()

    # Set HPC-specific parameter values
    args.accelerator = args.multi_gpu_backend
    args.gpus = args.num_gpus
    args.profiler = args.profiler_method

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
    model = LitClassifier(Backbone(hidden_dim=args.hidden_dim), args.num_epochs, args.lr)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.max_epochs = args.num_epochs

    # Resume from checkpoint if path to a valid one is provided
    args.ckpt_name = args.ckpt_name \
        if args.ckpt_name is not None \
        else 'LitClassifier-{epoch:02d}-{valid_cross_entropy:.2f}.ckpt'
    checkpoint_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    trainer.resume_from_checkpoint = checkpoint_path if os.path.exists(checkpoint_path) else None

    # Create and use callbacks
    early_stop_callback = EarlyStopping(monitor='valid_cross_entropy', mode='min', min_delta=0.00, patience=3)
    checkpoint_callback = ModelCheckpoint(monitor='valid_cross_entropy', save_top_k=3, dirpath=args.ckpt_dir,
                                          filename='LitClassifier-{epoch:02d}-{valid_cross_entropy:.2f}')
    lr_callback = LearningRateMonitor(logging_interval='epoch')  # Use with a learning rate scheduler
    trainer.callbacks = [early_stop_callback, checkpoint_callback, lr_callback]

    # Initialize logger
    args.experiment_name = f'LitClassifierWithBackbone-e{args.num_epochs}-b{args.batch_size}' \
        if not args.experiment_name \
        else args.experiment_name

    # Logging everything to Neptune
    # logger = NeptuneLogger(experiment_name=args.experiment_name if args.experiment_name else None,
    #                        project_name=args.project_name,
    #                        close_after_fit=False,
    #                        params={'max_epochs': args.num_epochs,
    #                                'batch_size': args.batch_size,
    #                                'lr': args.lr},
    #                        tags=['pytorch-lightning', 'image-classifier'],
    #                        upload_source_files=['*.py'])
    # logger.experiment.log_artifact(args.ckpt_dir)  # Neptune-specific

    # Logging everything to TensorBoard instead of Neptune
    # logger = TensorBoardLogger('tb_log', name=args.experiment_name)
    # trainer.logger = logger

    # Train with the provided model and data module
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)

    # logger.experiment.stop()  # Halt the current Neptune experiment


if __name__ == '__main__':
    cli_main()

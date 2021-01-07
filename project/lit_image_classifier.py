from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from comet_ml import API
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
    def __init__(self, backbone, learning_rate=1e-3):
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
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

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
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--early_stop_callback', type=bool, default=True)
    parser.add_argument('--num_dataloader_workers', type=int, default=1)
    parser.add_argument('--comet_workspace_name', type=str, default=None)
    parser.add_argument('--comet_project_name', type=str, default=None)
    parser.add_argument('--comet_model_name', type=str, default=None)
    parser.add_argument('--comet_registry_model_name', type=str, default=None)
    parser.add_argument('--comet_registry_model_version', type=str, default='1.0.0')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

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
    model = LitClassifier(Backbone(hidden_dim=args.hidden_dim), args.learning_rate)

    # ------------
    # checkpoint
    # ------------
    if args.comet_workspace_name is not None \
            and args.comet_project_name is not None \
            and args.comet_registry_model_name is not None:
        try:
            model = LitClassifier(Backbone(hidden_dim=args.hidden_dim), args.learning_rate)
            api = API()

            # Download an Experiment Model:
            if args.comet_model_name is not None:
                experiment = api.get(
                    f'{args.comet_workspace_name}/{args.comet_project_name}/{args.comet_experiment_name}')
                experiment.download_model(args.comet_model_name, output_path="./", expand=True)

            # Download a Registry Model:
            if args.comet_registry_model_name is not None:
                api.download_registry_model(args.comet_workspace_name, args.comet_registry_model_name,
                                            args.comet_registry_model_version, output_path="./", expand=True)

            # TODO: Research how to properly load a state dict file with PyTorch Lightning
            model = model.load_state_dict(
                torch.load(f'EarlyStopping-Adam-{args.batch_size}-{args.learning_rate}-Model.pth'))

            print('Resuming from checkpoint...')
        except AttributeError:
            print('Could not restore checkpoint. Skipping...')

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

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

    # ------------
    # finalizing
    # ------------
    torch.save(model.state_dict(), f'EarlyStopping-Adam-{args.batch_size}-{args.learning_rate}-Model.pth')
    comet_logger.experiment.log_model(f'EarlyStopping-Adam-{args.batch_size}-{args.learning_rate}-Model',
                                      f'EarlyStopping-Adam-{args.batch_size}-{args.learning_rate}-Model.pth')


if __name__ == '__main__':
    cli_main()

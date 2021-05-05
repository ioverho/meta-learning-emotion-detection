import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from transformers import AutoTokenizer
import argparse
import torch

from models.transformer_clf import Transformer_CLF
from data.go_emotions import go_emotions


class CLFTrainer(pl.LightningModule):
    
    def __init__(self, config):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        
        self.config = config
        # Create model        
        self.model = Transformer_CLF(config)
        # # Create loss module
        self.loss_module = nn.CrossEntropyLoss()


    def forward(self, text, attn_mask):
        return self.model(text, attn_mask)

        
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), self.config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]


    def encode(self, text, attn_mask=None):
        return self.model.encode(text, attn_mask)
    
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the train data loader.
        labels, text, attn_mask = batch
        preds = self.model(text, attn_mask)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log('train_acc', acc, on_step=False, on_epoch=True) # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_loss', loss)
        return loss # Return tensor to call ".backward" on


    def validation_step(self, batch, batch_idx):
        labels, text, attn_mask = batch
        preds = self.model(text, attn_mask).argmax(dim=-1)
        acc = (labels == preds).float().mean()

        self.log('val_acc', acc) # By default logs it per epoch (weighted average over batches)


    def test_step(self, batch, batch_idx):
        labels, text, attn_mask = batch
        preds = self.model(text, attn_mask).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('test_acc', acc) # By default logs it per epoch (weighted average over batches), and returns it afterwards


def train_model(config):
    """
    Function for training and testing a NLI model.
    Inputs:
        config - Namespace object from the argument parser
    """
    
    device = 'cuda' if (torch.cuda.is_available() and config['gpu']) else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(config['encoder_name'])
    
    # ToDo: process data and make sure it uses same amount of training data as protomaml
    dataset = go_emotions(first_label_only=True)
    dataset.prep()
    # train_loader, val_loader = dataset.get_dataloader('go_emotions', k=config["batch_size"],\
    #             tokenizer=tokenizer, shuffle=True)
    
    # ToDo: add n_classes
    config["n_classes"] = None


    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),                                  # Where to save models
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, 
                                                             mode="max", monitor="val_acc"), # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                         gpus=1 if str(device)=="cuda" else 0,                                                     # We run on a single GPU (if possible)
                         max_epochs=args.epochs,                                                                             # How many epochs to train for if no patience is set
                         callbacks=[LearningRateMonitor("epoch")],                                                   # Log learning rate every epoch
                         progress_bar_refresh_rate=100
                         )                                                                   # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    pl.seed_everything(args.seed) # To be reproducable
    
    model = CLFTrainer(config)
    trainer.fit(model, train_loader, val_loader)
    
    model = CLFTrainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    
    return model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## Dataset Initialization Hyperparameters
    parser.add_argument('--include', type=str, nargs='+', default=['grounded_emotions'],
                        help='Datasets to include.')

    parser.add_argument('--k', type=int, default=16,
                        help='The size of the k-shot. During training, also batch size.')

    ## Model Initialization Hyperparameters
    # Encoder
    parser.add_argument('--encoder_name', type=str, default='bert-base-uncased',
                        help='Pretrained encoder model matching import from Hugginface, e.g. "bert-base-uncased", "vinai/bertweet-base".')

    parser.add_argument('--nu', type=int, default=-1,
                        help='Max layer to keep frozen. 11 keeps enitre model frozen, -1 entirely trainable.')

    # Classifier
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='Hidden dimensions of the MLP. Pass a space separated list, e.g. "--hidden_dims 256 128".')

    parser.add_argument('--act_fn', type=str, default='Tanh',
                        help='Which activation to use. Currently either Tanh or ReLU.')
    



    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate for model update.')

    # parser.add_argument('--inner_lr', type=float, default=1e-3,
    #                     help='Learning rate for the task-specific model update.')

    # parser.add_argument('--output_lr', type=float, default=1e-3,
    #                     help='Learning rate for the softmax classification layer update.')

    parser.add_argument('--warmup_steps', type=float, default=250,
                        help='Learning warm-up steps for the shared model update. Uses linear schedule to constant.')

    # Hyperparams for training
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training and validation')

    # ## Meta-eval Hyperparameters

    # parser.add_argument('--n_eval_per_task', type=int, default=10,
    #                     help='Number of support sets to try for a single task.')

    # parser.add_argument('--n_eval_per_support', type=int, default=1,
    #                     help='Number of different batches to evaluate on per support set.')

    # parser.add_argument('--eval_every_n', type=int, default=125,
    #                     help='Number of different batches to evaluate on per support set.')

    ## MISC
    # Versioning, logging
    parser.add_argument('--version', type=str, default='grounded_emotions_test',
                        help='Construct model save name using versioning.')

    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/ProtoMAML",
                        help='Directory to save models to.')

    # Debugging
    parser.add_argument('--debug', default=False, type=lambda x: bool(strtobool(x)),
                        help=('Whether to run in debug mode'))
    parser.add_argument('--gpu', default=False, type=lambda x: bool(strtobool(x)),
                        help=('Whether to train on GPU (if available) or CPU'))

    config = vars(parser.parse_args())

    train_model(config)

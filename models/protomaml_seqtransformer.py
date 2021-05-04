from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.seqtransformer import SeqTransformer

class ProtoMAMLSeqTransformer(nn.Module):

    def __init__(self, config):
        """A transformer for sequence classification with the ability to train via FO-MAML

        Args:
            config (dict): dictionary with corresponding args.
        """
        super().__init__()

        self.model_shared = SeqTransformer(config)
        self.model_task = deepcopy(self.model_shared)

        self.n_inner = config['n_inner']
        self.inner_lr = config['inner_lr']
        self.output_lr = config['output_lr']
        self.lossfn = config['lossfn']()

        self.task_name = None
        self.W_task = None
        self.b_task = None

    def train(self):
        """Set model(s) to train.
        """

        self.model_shared.train()
        self.model_task.train()

    def eval(self):
        """Set model(s) to eval.
        """

        self.model_shared.eval()
        self.model_task.eval()

    def device(self):
        """
        Hacky method for checking model device.
        Requires all parameters to be on same device.
        """
        return next(self.model_shared.parameters()).device

    def forward(self, text, attn_mask=None):
        """
        Task-specific classification of a sequence.
        For safety, will not classify without initial adaptation, but otherwise will classify with whatever
        the current task-specific parameters are.

        Args:
            labels (LongTensor): batch labels
            text (LongTensor): batch text, converted to interger Tensor
            attn_mask LongTensor, optional): attention mask, values from {0, 1}. Defaults to None.

        Returns:
            Tensor: tensor with logits.
        """

        if self.W_task == None or self.b_task == None:
            raise ValueError('No task-specific model specified yet.')

        y = self.model_task(text, attn_mask)
        logits = F.linear(y, self.W_task, self.b_task)

        return logits

    def _generate_protoypes(self, labels, text, attn_mask=None):

        y = self.model_shared(text, attn_mask)

        prototypes = [torch.mean(y[labels == i], dim=0)
                      for i in torch.unique(labels)]
        prototypes = torch.stack(prototypes)

        return prototypes

    def generate_clf_weights(self, labels, text, attn_mask=None):
        """Generates the classification layer weights from prototypes derived from a single batch.

        Args:
            labels (LongTensor): batch labels
            text (LongTensor): batch text, converted to interger Tensor
            attn_mask LongTensor, optional): attention mask, values from {0, 1}. Defaults to None.

        Returns:
            tuple of tensors: first tensor are weights, second biases
        """

        prototypes = self._generate_protoypes(labels, text, attn_mask)

        W_init = 2 * prototypes
        b_init = -torch.norm(prototypes, p=2, dim=1)

        return W_init, b_init

    def adapt(self, labels, text, attn_mask=None, task_name=None, verbose=False):
        """Perform MAML adaption with Prototypical initialization of classification layer.

        Args:
            labels (LongTensor): batch labels
            text (LongTensor): batch text, converted to interger Tensor
            attn_mask LongTensor, optional): attention mask, values from {0, 1}. Defaults to None.
            task_name (str, optional): name of current task for administration within model. Defaults to None.
            verbose (bool, optional): whether or not to print inner loop updates. Defaults to False.
        """

        self.task_name = task_name

        # Clone model for task specific episode model
        self.model_task = deepcopy(self.model_shared)

        task_optimizer = optim.SGD(
            self.model_task.parameters(), lr=self.inner_lr)

        # Generate initial classification weights
        W_init, b_init = self.generate_clf_weights(labels, text, attn_mask)

        # Detach the initial weights from task-specific model
        self.W_task, self.b_task = W_init.detach().clone(), b_init.detach().clone()
        self.W_task.requires_grad, self.b_task.requires_grad = True, True

        output_optimizer = optim.SGD(
            [self.W_task, self.b_task], lr=self.output_lr)

        for i in range(self.n_inner):

            # Embed, encode, classify and compute loss
            logits = self.forward(text, attn_mask)
            loss = self.lossfn(logits, labels)

            # Backprop the output parameters
            # Retrain graph for shared parameters
            self.W_task.grad, self.b_task.grad = torch.autograd.grad(
                loss, [self.W_task, self.b_task], retain_graph=True)

            # Calculate the gradients on shared parameters here
            updateable_task_params = [
                param for param in self.model_task.parameters() if param.requires_grad]
            task_grads = torch.autograd.grad(loss, updateable_task_params)

            # Store task-specific gradients
            for param, grad in zip(updateable_task_params, task_grads):
                param.grad = grad

            # Update the parameters
            output_optimizer.step()
            task_optimizer.step()

            output_optimizer.zero_grad()
            task_optimizer.zero_grad()

            if verbose:
                print("\tInner {} | Loss {:.4E}".format(
                    i, loss.detach().item()))

        self.W_task = W_init + (self.W_task - W_init).detach()
        self.b_task = b_init + (self.b_task - b_init).detach()

    def backward(self, loss):
        """Backpropagate a loss on the task-specific model to the shared model parameters

        Args:
            loss (Tensor)
        """

        # Calculate gradients for task-specific parameters
        updateable_task_params = [
            param for param in self.model_task.parameters() if param.requires_grad]
        task_grads = torch.autograd.grad(
            loss, updateable_task_params, retain_graph=True)
        updateable_task_params = None

        # Calculate gradients for shared model parameters
        updateable_shared_params = [
            param for param in self.model_shared.parameters() if param.requires_grad]
        shared_grads = torch.autograd.grad(loss, updateable_shared_params)

        # Accumulate gradients
        for param, g_task, g_shared in zip(updateable_shared_params, task_grads, shared_grads):
            if param.grad == None:
                param.grad = g_shared + g_task
            else:
                param.grad += g_shared + g_task
        updateable_shared_params = None

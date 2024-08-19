import os
import numpy as np
import torch
import torch.nn as nn
import copy
import re
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam, RMSprop, SGD # type: ignore
import torch.nn.functional as F
from pathlib import Path
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import plot_output, plot_output_recon, weights_heatmap
from models.inception_unet import InceptionUNet
from models.ad_tfm import AD_TFM
from models.inception_unet_old import InceptionUNet as InceptionUNetOld


def train(model, dataloader, criterions, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        target = target[:, -1, :]
        # output = torch.mean(output.squeeze(-1), dim=-1).reshape(-1, 1)

        loss = criterions(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss

# Define the evaluation loop
def evaluate(model, dataloader, criterions, device):
    model.eval()
    running_loss = 0.0

    with torch.inference_mode():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            target = target[:, -1, :]
            # output = torch.mean(output.squeeze(-1), dim=-1).reshape(-1, 1)

            loss = criterions(output, target)
            running_loss += loss.item() * data.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss

def train_recon(model, dataloader, criterions, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)

        output = model(data, data)

        loss = criterions(output, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss

# Define the evaluation loop
def evaluate_recon(model, dataloader, criterions, device):
    model.eval()
    running_loss = 0.0

    with torch.inference_mode():
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)

            output = model(data, data)

            loss = criterions(output, data)
            running_loss += loss.item() * data.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss

def train_and_evaluate(model, train_loaders, val_loaders, criterions, optimizer, learning,
                       num_epochs, device, scheduler=None, scheduler_type=None, es=None, tensorboard_writer=None):# scheduler_type, 

    # Create empty results dictionary
    results = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    best_model_weights = model.state_dict()

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_val_loss = 0.0

        if learning == "sup":

            for train_loader in train_loaders:
                train_loss = train(model, train_loader, criterions, optimizer, device)
                total_train_loss += train_loss

            for val_loader in val_loaders:
                val_loss = evaluate(model, val_loader, criterions, device)
                total_val_loss += val_loss

        elif learning == "unsup":

            for train_loader in train_loaders:
                train_loss = train_recon(model, train_loader, criterions, optimizer, device)
                total_train_loss += train_loss

            for val_loader in val_loaders:
                val_loss = evaluate_recon(model, val_loader, criterions, device)
                total_val_loss += val_loss

        avg_train_loss = total_train_loss / len(train_loaders)
        avg_val_loss = total_val_loss / len(val_loaders)

        tensorboard_writer.add_scalar('Loss/train', avg_train_loss, epoch) # type: ignore # type: ignore
        tensorboard_writer.add_scalar('Loss/val', avg_val_loss, epoch) # type: ignore
        # if epoch % 10 == 0:
        #     for name, param in model.named_parameters():
        #         tensorboard_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch) # type: ignore

        tensorboard_writer.close() # type: ignore # type: ignore
        if scheduler is not None:
            scheduler.step(avg_val_loss) if scheduler_type=='plateau' else scheduler.step()

        # Print out what is happening
        if es(model, avg_val_loss): # type: ignore
            print(
                f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.8f}, '
                f'Val Loss: {avg_val_loss:.8f}') # type: ignore
            if epoch > 20:
                break
        else:
            print(
                f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.8f}, '
                f'Val Loss: {avg_val_loss:.8f}, EStop:[{es.status}]') # type: ignore
                
            # Update results dictionary
            results["train_loss"].append(avg_train_loss)
            results["val_loss"].append(avg_val_loss)
            
        # Save the best model weights
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict()

    # Load the best model weights
    model.load_state_dict(best_model_weights)
    print(best_val_loss)          
    return results


def kl_mse_loss(output, target, alpha=0.5):
    # Ensure target is probability distribution over features (softmax)
    target_prob = F.softmax(target, dim=-1)  # Apply softmax across features

    # Ensure output is log-probability distribution over features (log-softmax)
    output_log_prob = F.log_softmax(output, dim=-1)  # Apply log-softmax across features

    # Calculate KL divergence for each element in the sequence and batch, then average
    kl = F.kl_div(output_log_prob, target_prob, reduction='batchmean')

    # MSE loss, considering all elements in the batch and sequence
    mse = F.mse_loss(output, target, reduction='mean')

    # Combined loss with weighting
    loss = alpha * mse + (1 - alpha) * kl
    return loss


def best_weights(model, val_loss):
    best_model = copy.deepcopy(model)
    if best_loss > val_loss: # type: ignore
        best_loss = val_loss
        best_model.load_state_dict(model.state_dict())
        best_model = copy.deepcopy(model)
    return best_model

def select_model(model_type, input_size, emb_size, seq_len, hidden_size, 
                 enc_layers, dec_layers, num_heads, dropout, batch_size, device):
    
    if model_type == 'incept':
        return InceptionUNetOld(in_channels=seq_len, input_size=input_size, emb_size=emb_size, kernel_size=3, stride=1, dropout=0.1).to(device)
    
    elif model_type == 'tfm':
        return AD_TFM(d_model=input_size, emb_size=emb_size, nhead=num_heads, seq_len=seq_len, num_encoder_layers=enc_layers,
                      num_decoder_layers=dec_layers, dim_feedforward=hidden_size, activation=nn.ReLU(), dropout=dropout,
                      norm_first=False).to(device)
    # elif model_type == 'tf':
    #     return TimeSeriesTransformer(num_features=seq_len, d_model=emb_size, nhead=num_heads, num_encoder_layers=enc_layers,
    #                                  num_decoder_layers=dec_layers, dim_feedforward=seq_len*2, seq_len=input_size, dropout=dropout).to(device)

    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
def scheduler_lr(type, optimizer, max_lr, min_lr):
     if type == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=min_lr, last_epoch=-1) # type: ignore
     elif type == 'cyclic':
        return lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=5, step_size_down=None, 
                                     gamma=1.0, cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
     elif type == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, threshold=0.0001,  # type: ignore
                                              threshold_mode='rel', cooldown=5, min_lr=1e-06, eps=1e-08)
     elif type == 'multistep':
        return lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25], gamma=0.1, last_epoch=-1) # type: ignore
     else:
        raise ValueError(f"Invalid scheduler type: {type}")
     
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict()) # type: ignore
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = f"Stopped on {self.counter}"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict()) # type: ignore
                return True
        self.status = f"{self.counter}/{self.patience}"
        return False

def save_model(model, results, target_dir, model_name, time):
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    model_state_dict = model.state_dict()
    save_dict = {'model_state_dict': model_state_dict,
                 'results': results,
                 'time': time}
    torch.save(obj=save_dict,
               f=model_save_path)

# Prediction utils
def load_model(model_variable, model_dir, trained_model_name, device):

    # Load the model
    model_dict = torch.load(f"{model_dir}/{trained_model_name}", map_location=torch.device(device), weights_only=False)
    model_state_dict = model_dict['model_state_dict']
    model_variable.load_state_dict(model_state_dict)
    
    return model_variable


# define the adam optimizer
def adam_optimizer(model, lr, betas, eps, weight_decay, amsgrad):
    optimizer = Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    return optimizer

def rmsprop_optimizer(model, lr, alpha, eps, weight_decay, momentum, centered):
    optimizer = RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                              momentum=momentum, centered=centered)
    return optimizer

def sgd_optimizer(model, lr, momentum, dampening, weight_decay, nesterov):
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                          nesterov=nesterov)
    return optimizer


# Infer utils
def infer(model, test_loaders, criterion, kalman_params, plot_name, device):
    model.eval()
    running_loss, total_train_loss = 0.0, 0.0
    targets = []
    outputs = []

    with torch.inference_mode():
        for dataloader in test_loaders:    
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                target = target[:, -1, :]
                # output = torch.mean(output.squeeze(-1), dim=-1).reshape(-1, 1)

                targets.extend(target.cpu().numpy().tolist())
                outputs.extend(output.cpu().numpy().tolist())

    targets = np.array(targets).reshape(-1,)
    outputs = np.array(outputs).reshape(-1,)

    targets, outputs = plot_output(targets, outputs, plot_name)
    return np.array(targets), np.array(outputs)


def infer_recon(model, test_loaders, device):
    model.eval()
    running_loss, total_train_loss = 0.0, 0.0
    output_dict = {}
    loss = []

    with torch.inference_mode():
        for dataloader in test_loaders:    
            for batch_idx, data in enumerate(dataloader):
                data = data.to(device)
                output = model(data, data)
                loss.append(np.mean(np.abs(output.cpu().numpy() - data.cpu().numpy()), axis=1))

                output_dict[batch_idx] = {"inputs": data.cpu().numpy(), "outputs": output.cpu().numpy()}
    
    # Concatenate the inputs and outputs into single arrays
    inputs_concat = np.concatenate([v['inputs'] for v in output_dict.values()], axis=0)
    outputs_concat = np.concatenate([v['outputs'] for v in output_dict.values()], axis=0)
    output_dic = {"inputs": inputs_concat, "outputs": outputs_concat}
    loss = np.concatenate(loss, axis=0)

    return output_dic, loss

def extracting_parameters(model_name):
    parameters = []
    pattern = r"[-+]?\d*\.\d+|\d+"
    
    for item in model_name:
        matches = re.findall(pattern, item)
        for num in matches:
            parameters.append(float(num) if '.' in num else int(num))
    
    return parameters

class ParseBool:
    """
    Class to parse a string value into a boolean or Null value
    """
    def __init__(self, value):
        self.value = self.parse_value(value)

    def parse_value(self, value):
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, str):
            if value.lower() == 'true':
                return True
            elif value.lower() == 'false':
                return False
        return None

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, ParseBool):
            return self.value == other.value
        return self.value == self.parse_value(other)

    def __repr__(self):
        return f"ParseBool({self.value})"

    def __bool__(self):
        return self.value is not None and self.value is True
    
def parse_bool(value):
    return ParseBool(value).value


def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from models.badnet import BadNet
from models.LoRA import LoRAConfig

def array2img(x):
    a = np.array(x)
    img = Image.fromarray(a.astype('uint8'), 'RGB')
    img.show()


def print_model_perform(model, data_loader):
    model.eval() # switch to eval mode
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        
        batch_y = torch.argmax(batch_y, dim=1)
        y_true.append(batch_y)
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)

    try:
        target_names_idx = set.union(set(np.array(y_true.cpu())), set(np.array(y_predict.cpu())))
        target_names = [data_loader.dataset.classes[i] for i in target_names_idx]
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=target_names))
    except ValueError as e:
        print(e)


def loss_picker(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("automatically assign MSE loss function to you...")
        criterion = nn.MSELoss()
    return criterion


def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)
    return optimizer


def backdoor_model_trainer(dataname, train_data_loader, test_data_ori_loader, test_data_tri_loader, trigger_label, epoch, batch_size, loss_mode, optimization, lr, print_perform_every_epoch, basic_model_path, device):
    rank = 16
    lora_alpha = 16.0
    lora_dropout = 0.15
    freeze_weights = True

    lora_config = LoRAConfig(rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, freeze_weights=freeze_weights)


    badnet = BadNet(
        input_channels=train_data_loader.dataset.channels, 
        output_num=train_data_loader.dataset.class_num, 
        config=lora_config  
    ).to(device)

    criterion = loss_picker(loss_mode)
    # optimizer = optimizer_picker(optimization, badnet.parameters(), lr=lr)
    lora_parameters = [p for p in badnet.parameters() if p.requires_grad]  # Only train LoRA parameters
    optimizer = optimizer_picker(optimization, lora_parameters, lr=lr)



    train_process = []
    print("### target label is %d, EPOCH is %d, Learning Rate is %f" % (trigger_label, epoch, lr))
    print("### Train set size is %d, ori test set size is %d, tri test set size is %d\n" % (len(train_data_loader.dataset), len(test_data_ori_loader.dataset), len(test_data_tri_loader.dataset)))
    try:
        for epo in range(epoch):
            loss = train(badnet, train_data_loader, criterion, optimizer, loss_mode)
            acc_train = eval(badnet, train_data_loader, batch_size=batch_size, mode='backdoor', print_perform=print_perform_every_epoch)
            acc_test_ori = eval(badnet, test_data_ori_loader, batch_size=batch_size, mode='backdoor', print_perform=print_perform_every_epoch)
            acc_test_tri = eval(badnet, test_data_tri_loader, batch_size=batch_size, mode='backdoor', print_perform=print_perform_every_epoch)

            print("# EPOCH%d   loss: %.4f  training acc: %.4f, ori testing acc: %.4f, trigger testing acc: %.4f\n"\
                % (epo, loss.item(), acc_train, acc_test_ori, acc_test_tri))
            
            # save model to checkpoints directory
            torch.quantization.convert(badnet, inplace=True)
            torch.save(badnet.state_dict(), basic_model_path)

            # save training progress
            train_process.append(( dataname, batch_size, trigger_label, lr, epo, loss.item(), acc_train, acc_test_ori, acc_test_tri))
            df = pd.DataFrame(train_process, columns=("dataname", "batch_size", "trigger_label", "learning_rate", "epoch", "loss", "train_acc", "test_ori_acc", "test_tri_acc"))
            df.to_csv("./logs/%s_train_process_trigger%d.csv" % (dataname, trigger_label), index=False, encoding='utf-8')
    except KeyboardInterrupt:
        show_graphs(lora_config=lora_config, train_process=train_process)
        return badnet

    show_graphs(lora_config=lora_config, train_process=train_process)


    return badnet

def show_graphs(lora_config, train_process):
    print("Plotting results...")
    _, _, _, _, epochs, losses, train_accs, ori_test_accs, trigger_test_accs = zip(*train_process)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label="Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Training Accuracy", color="blue")
    plt.plot(epochs, ori_test_accs, label="Original Test Accuracy", color="green")
    plt.plot(epochs, trigger_test_accs, label="Trigger Test Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracies Over Epochs")
    plt.legend()
    lora_text = (
        f"LoRA Config:\n"
        f"Rank: {lora_config.rank}\n"
        f"Alpha: {lora_config.lora_alpha}\n"
        f"Dropout: {lora_config.lora_dropout}\n"
        f"Freeze Weights: {lora_config.freeze_weights}"
    )
    plt.gcf().text(0.75, 0.3, lora_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    plt.show()

def train(model, data_loader, criterion, optimizer, loss_mode):
    running_loss = 0
    model.train()
    
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x

        if loss_mode == "mse":
            loss = criterion(output, batch_y) # mse loss
        elif loss_mode == "cross":
            loss = criterion(output, torch.argmax(batch_y, dim=1)) # cross entropy loss

        loss.backward()
        optimizer.step()
        running_loss += loss
    return running_loss


def eval(model, data_loader, batch_size=64, mode='backdoor', print_perform=False):
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        
        batch_y = torch.argmax(batch_y, dim=1)
        y_true.append(batch_y)
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)

    if print_perform and mode is not 'backdoor':
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return accuracy_score(y_true.cpu(), y_predict.cpu())
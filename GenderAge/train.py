from tqdm import tqdm
from torch import nn

from .utils import *
from .eval import *
from .model import * 

import os 
import torch

def train(train_data_dir, weights, device = 'cpu', image_size = 64, batch_size = 128, num_epochs = 50, steps_per_epoch = None,
              val_data_dir = None, validation_split = None, save_history = True):
    
    # đặt val_data_dir and validation_split không đồng thời khác None
    assert not(val_data_dir is not None and validation_split is not None)

    if isinstance(device, str):
        if (device == 'cuda' or device == 'gpu') and torch.cuda.is_available():
            device = torch.device(device)
        else:
            device = torch.device('cpu')

    # add model to device
    model = GenderAgePrediction()
    model = model.to(device)

    if weights and os.path.exists(weights):
        model.load_state_dict(torch.load(weights))
        print('Weights loaded successfully from path:', weights)
        print('====================================================')

    # get train_loader
    train_data = get_dataloader(train_data_dir, image_size = image_size, batch_size = batch_size)

    # chia dữ liệu thành 2 tập train và val    
    if val_data_dir is not None:
        val_data = get_dataloader(val_data_dir, image_size = image_size, batch_size = batch_size)
    elif validation_split is not None: 
        train_data, val_data = split_dataloader(train_data, validation_split)
    else: 
        val_data = None 

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)

    # 
    num_steps = len(train_data)
    iterator = iter(train_data)
    count_steps = 1

    # history
    history = {
        'train_gender_loss': [],
        'train_age_loss': [],
        'train_gender_acc': [],
        'train_age_acc': [],
        'val_gender_loss': [],
        'val_age_loss': [],
        'val_gender_acc': [],
        'val_age_acc': [],
    }

    # Loss function
    AgeLoss = nn.MSELoss()
    GenderLoss = nn.BCELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    
    for epoch in range(1, num_epochs + 1):
        total_gender_loss = 0
        total_age_loss = 0 
        age_accuracy = 0
        gender_accuracy = 0
        
        model.train()
        for step in tqdm(range(steps_per_epoch), desc = f'Epoch {epoch}/{num_epochs}: ', ncols = 100):
            images, genders, ages = next(iterator)
            batch_size = images.shape[0]
            
            images, genders, ages = images.to(device), genders.to(device), ages.to(device)
            
            pred_genders, pred_ages = model(images)
            pred_genders, pred_ages = pred_genders.view(batch_size), pred_ages.view(batch_size)
                        
            age_loss = AgeLoss(pred_ages, ages.float())
            gender_loss = GenderLoss(pred_genders, genders.float())
            
            loss = age_loss + gender_loss
            
            total_age_loss += age_loss.item()
            total_gender_loss += gender_loss.item()
            
            gender_acc = torch.sum(torch.round(pred_genders) == genders)/batch_size
            age_acc = torch.sum(torch.round(pred_ages) == ages)/batch_size
            
            age_accuracy += age_acc.item()
            gender_accuracy += gender_acc.item()
            
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # nếu nó duyệt hết qua tập dữ liệu thì cho nó lặp lại 1 lần nữa
            if count_steps == num_steps:
                iterator = iter(train_data)
                count_steps = 0
            count_steps += 1
            
        train_age_loss = total_age_loss / steps_per_epoch
        train_gender_loss = total_gender_loss / steps_per_epoch
        train_age_accuracy = age_accuracy / steps_per_epoch
        train_gender_accuracy = gender_accuracy / steps_per_epoch

        history['train_age_loss'].append(float(train_age_loss))
        history['train_gender_loss'].append(float(train_gender_loss))
        history['train_age_acc'].append(float(train_age_accuracy))
        history['train_gender_acc'].append(float(train_gender_accuracy))

        print(f'Epoch: {epoch}, train_age_loss: {train_age_loss: .2f}, train_gender_loss: {train_gender_loss: .3f}, train_age_accuracy: {train_age_accuracy: .2f}, train_gender_accuracy: {train_gender_accuracy: .2f}')
        if val_data:
            val_age_loss, val_gender_loss, val_age_accuracy, val_gender_accuracy = evaluate(model, val_data, device = device)
            history['val_age_loss'].append(float(val_age_loss))
            history['val_gender_loss'].append(float(val_gender_loss))
            history['val_age_acc'].append(float(val_age_accuracy))
            history['val_gender_acc'].append(float(val_gender_accuracy))
            
            print(f'Epoch: {epoch}, val_age_loss: {val_age_loss: .2f}, val_gender_loss: {val_gender_loss: .3f}, val_age_accuracy: {val_age_accuracy: .2f}, val_gender_accuracy: {val_gender_accuracy: .2f}')

    if weights:  
        torch.save(model.state_dict(), weights)
        print(f'Saved successfully last weights to:', weights)
        
    if save_history:
        visualize_history(history, save_history)

if __name__ == '__main__':
    batch_size = 128
    image_size = 64
    train_data_dir = '/kaggle/input/utkface-new/UTKFace'
    device = 'cuda'
    weights = 'weights\AgeGenderWeights.pt'
    epochs = 100
    train(train_data_dir, weights, device = device, steps_per_epoch = None,
          validation_split = 0.2, image_size = image_size, batch_size = batch_size, save_history = True)
    
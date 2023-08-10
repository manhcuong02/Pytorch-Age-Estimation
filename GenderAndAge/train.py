from tqdm import tqdm
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils import *
from .eval import *
from .models import * 

import os 
import torch
import numpy as np


def train(train_data_dir, weights, device = 'cpu', image_size = 64, batch_size = 128, num_epochs = 100, steps_per_epoch = None,
              val_data_dir = None, validation_split = None, save_history = True):
    
    # đặt val_data_dir and validation_split không đồng thời khác None
    assert not(val_data_dir is not None and validation_split is not None)

    if isinstance(device, str):
        if (device == 'cuda' or device == 'gpu') and torch.cuda.is_available():
            device = torch.device(device)
        else:
            device = torch.device('cpu')

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

    # steps per epoch
    num_steps = len(train_data)
    iterator = iter(train_data)
    count_steps = 1

    # add model to device
    gender_model = GenderClassificationModel().to(device)
    age_range_model = AgeRangeModel().to(device)
    age_estimation_model = AgeEstimationModel().to(device)

    # Loss function
    AgeRangeLoss = nn.CrossEntropyLoss()
    GenderLoss = nn.BCELoss()
    AgeEstimationLoss = nn.L1Loss()
    
    # Optimizer
    gen_optimizer = torch.optim.Adam(gender_model.parameters(), lr = 1e-4)
    age_range_optimizer = torch.optim.Adam(age_range_model.parameters(), lr = 5e-3)
    age_estimation_optimizer = torch.optim.Adam(age_estimation_model.parameters(), lr = 1e-3)

    #schedular
    age_range_scheduler = ReduceLROnPlateau(age_range_optimizer, mode = 'min', factor = 0.1, patience = 3, verbose = 1)
    gender_scheduler = ReduceLROnPlateau(gen_optimizer, mode = 'min', factor = 0.1, patience = 3, verbose = 1)
    age_estimation_scheduler = ReduceLROnPlateau(age_estimation_optimizer, mode = 'min', factor = 0.1, patience = 3, verbose = 1)

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
    
    for epoch in range(1, num_epochs + 1):
        total_gender_loss = 0
        total_age_loss = 0 
        age_accuracy = 0
        gender_accuracy = 0
        
        total_age_estimation_loss = 0
        
        gender_model.train()
        age_range_model.train()
        age_estimation_model.train()
        
        for step in tqdm(range(steps_per_epoch), desc = f'Epoch {epoch}/{num_epochs}: ', ncols = 100):
            images, genders, age_labels, ages = next(iterator)
            batch_size = images.shape[0]
            
            images, genders, age_labels, ages = images.to(device), genders.to(device), age_labels.to(device), ages.to(device)
            
            pred_genders = gender_model(images).view(-1)
            pred_age_labels = age_range_model(images)
            
            age_loss = AgeRangeLoss(pred_age_labels, age_labels.long())
            
            gender_loss = GenderLoss(pred_genders, genders.float())
                        
            total_age_loss += age_loss.item()
            total_gender_loss += gender_loss.item()
            
            gender_acc = torch.sum(torch.round(pred_genders) == genders)/batch_size
            age_acc = torch.sum(torch.argmax(pred_age_labels, dim = 1) == age_labels)/batch_size
            
            age_accuracy += age_acc
            gender_accuracy += gender_acc
            
            age_range_optimizer.zero_grad()
            age_loss.backward()
            age_range_optimizer.step()
            
            gen_optimizer.zero_grad()
            gender_loss.backward()
            gen_optimizer.step()
            
            # age estimation loss
            estimated_ages = age_estimation_model(images, age_labels).view(-1)
            age_estimation_loss = AgeEstimationLoss(ages, estimated_ages)
            
            age_estimation_optimizer.zero_grad()
            age_estimation_loss.backward()
            age_estimation_optimizer.step()
            
            total_age_estimation_loss += age_estimation_loss.item()
            
            # nếu nó duyệt hết qua tập dữ liệu thì cho nó lặp lại 1 lần nữa
            if count_steps == num_steps:
                iterator = iter(train_data)
                count_steps = 0
            count_steps += 1
            
        train_age_loss = total_age_loss / steps_per_epoch
        train_gender_loss = total_gender_loss / steps_per_epoch
        train_age_accuracy = age_accuracy / steps_per_epoch
        train_gender_accuracy = gender_accuracy / steps_per_epoch

        train_age_estimation_loss = total_age_estimation_loss/steps_per_epoch
        
        history['train_age_loss'].append(float(train_age_loss))
        history['train_gender_loss'].append(float(train_gender_loss))
        history['train_age_acc'].append(float(train_age_accuracy))
        history['train_gender_acc'].append(float(train_gender_accuracy))

        print(f'train_age_loss: {train_age_loss: .2f}, train_gender_loss: {train_gender_loss: .3f}, train_age_accuracy: {train_age_accuracy: .2f}, train_gender_accuracy: {train_gender_accuracy: .2f}, train_age_estimation_loss: {train_age_estimation_loss: .3f}')
        if val_data:
            val_age_loss, val_gender_loss, val_age_accuracy, val_gender_accuracy, val_age_estimation_loss = evaluate(gender_model, age_range_model, age_estimation_model, val_data, device = device)
            history['val_age_loss'].append(float(val_age_loss))
            history['val_gender_loss'].append(float(val_gender_loss))
            history['val_age_acc'].append(float(val_age_accuracy))
            history['val_gender_acc'].append(float(val_gender_accuracy))
            
            age_range_scheduler.step(np.round(val_age_loss, 3))
            gender_scheduler.step(np.round(val_gender_loss, 3))
            age_estimation_scheduler.step(np.round(val_age_estimation_loss, 3))
            print(f'val_age_loss: {val_age_loss: .2f}, val_gender_loss: {val_gender_loss: .3f}, val_age_accuracy: {val_age_accuracy: .2f}, val_gender_accuracy: {val_gender_accuracy: .2f}, val_age_estimation_loss: {val_age_estimation_loss : .3f}')

    if weights:  
        class dummy_model(nn.Module):
            def __init__(self):
                super().__init__()
                
                self.gender_model = gender_model
                self.age_range_model = age_range_model
                self.age_estimation_model = age_estimation_model
            
            def forward(self, x):
                return
        
        model = dummy_model()
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
    
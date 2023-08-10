from torch import nn
from tqdm import tqdm

import torch

def evaluate(gender_model, age_range_model, age_estimation_model, val_dataloader, device = 'cpu', verbose = 0):
    # set device
    if isinstance(device, str):
        if (device == 'cuda' or device == 'gpu') and torch.cuda.is_available():
            device = torch.device(device)
        else:
            device = torch.device('cpu')

    # Loss function
    AgeRangeLoss = nn.CrossEntropyLoss()
    GenderLoss = nn.BCELoss()
    AgeEstimationLoss = nn.L1Loss()
    
    gender_model = gender_model.to(device)
    age_range_model = age_range_model.to(device)
    age_estimation_model = age_estimation_model.to(device)
    
    with torch.no_grad():
        age_range_model.eval()
        gender_model.eval()
        age_estimation_model.eval()
        age_accuracy = 0
        gender_accuracy = 0
        total_age_loss = 0
        total_gender_loss = 0
        total_age_estimation_loss = 0
        if verbose == 1:
            val_dataloader = tqdm(val_dataloader, desc = 'Evaluate: ', ncols = 100)
            
        for images, genders, age_labels, ages in val_dataloader:
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
            
            estimated_ages  = age_estimation_model(images, age_labels).view(-1)
            age_estimation_loss = AgeEstimationLoss(ages, estimated_ages)
            
            total_age_estimation_loss += age_estimation_loss.item()
            
        val_age_loss = total_age_loss / len(val_dataloader)
        val_gender_loss = total_gender_loss / len(val_dataloader)
        val_age_accuracy = age_accuracy / len(val_dataloader)
        val_gender_accuracy = gender_accuracy / len(val_dataloader)
        val_age_estimation_loss = total_age_estimation_loss / len(val_dataloader)
        
        return val_age_loss, val_gender_loss, val_age_accuracy, val_gender_accuracy, val_age_estimation_loss

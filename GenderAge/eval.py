from torch import nn
from tqdm import tqdm

import torch

def evaluate(model, val_dataloader, device = 'cpu', weights = None, verbose = 0):
    # set device
    if isinstance(device, str):
        if (device == 'cuda' or device == 'gpu') and torch.cuda.is_available():
            device = torch.device(device)
        else:
            device = torch.device('cpu')

    if weights:
        model.load_state_dict(torch.load(weights))
        print('Weights loaded successfully from path:', weights)
        print('====================================================')

    # Loss function
    AgeLoss = nn.MSELoss()
    GenderLoss = nn.BCELoss()
        
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        age_accuracy = 0
        gender_accuracy = 0
        total_age_loss = 0
        total_gender_loss = 0
        if verbose == 1:
            val_dataloader = tqdm(val_dataloader, desc = 'Evaluate: ', ncols = 100)
            
        for images, genders, ages in val_dataloader:
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
            
            age_accuracy += age_acc
            gender_accuracy += gender_acc
            
        average_age_loss = total_age_loss / len(val_dataloader)
        average_gender_loss = total_gender_loss / len(val_dataloader)
        age_accuracy = age_accuracy / len(val_dataloader)
        gender_accuracy = gender_accuracy / len(val_dataloader)
        
        return average_age_loss, average_gender_loss, age_accuracy, gender_accuracy

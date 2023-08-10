from torch import nn
import torch
from torch.nn import functional as F

class GenderClassificationModel(nn.Module):
    "VGG-Face"
    def __init__(self):
        super().__init__()
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(2048, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1)
        
    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        return F.sigmoid(self.fc8(x))
    
class AgeRangeModel(nn.Module):
    def __init__(self, in_channels = 3, backbone = 'resnet50', pretrained = False, num_classes = 9):
        super().__init__()
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),             
        )
    
        self.Conv2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2), 
        )
        
        self.Conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2), 
        )
        
        self.adap = nn.AdaptiveAvgPool2d((2,2))
        
        self.out_age = nn.Sequential(
            nn.Linear(2048, num_classes)
#             nn.Softmax(dim = 1)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        
        x = self.adap(x)
        
        x = x.view(batch_size, -1)
        
        x = self.out_age(x)
    
        return x

class AgeEstimationModel(nn.Module):
    "VGG-Face"
    def __init__(self):
        super().__init__()
        self.embedding_layer = nn.Embedding(9, 64)
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),             
        )
    
        self.Conv2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2), 
        )
        
        self.Conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2), 
        )
        
        self.adap = nn.AdaptiveAvgPool2d((2,2))
        
        self.out_age = nn.Sequential(
            nn.Linear(2048 + 64, 1),
            nn.ReLU()
        )
        
    def forward(self, x, y):
        batch_size = x.shape[0]
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        
        x = self.adap(x)
        
        x = x.view(batch_size, -1)
        
        y = self.embedding_layer(y)
        
        x = torch.cat([x,y], dim = 1)
        
        x = self.out_age(x)
    
        return x
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.gender_model = GenderClassificationModel()
        
        self.age_range_model = AgeRangeModel()
        
        self.age_estimation_model = AgeEstimationModel()

    def forward(self,x):
        """x: batch, 3, 64, 64"""
        if len(x.shape) == 3:
            x = x[None, ...]
        
        predicted_genders = self.gender_model(x)
        
        age_ranges = self.age_range_model(x)
        
        y = torch.argmax(age_ranges, dim = 1).view(-1,)
        
        estimated_ages = self.age_estimation_model(x, y)
        
        return predicted_genders, estimated_ages
    
if __name__ == '__main__':
    model = Model()
    x = torch.rand(2,3,64,64)
    genders, ages = model(x) #
    
    print(genders.shape, ages.shape)
    
        
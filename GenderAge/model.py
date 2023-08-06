from torch import nn
import timm
import torch

class GenderAgePrediction(nn.Module):
    def __init__(self, in_channels = 3, backbone = 'resnet101', pretrained = False):
        super().__init__()
        self.pretrained = pretrained
        self.backbone = timm.create_model(backbone, features_only=True, pretrained = pretrained)
        
        self.adap = nn.AdaptiveAvgPool2d(1)
        
        self.out_gender = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
        
        self.out_age = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        
    def forward(self, x):
        if self.training and self.pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        batch_size = x.shape[0]
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        x = self.adap(x5)
        
        x = x.view(batch_size, -1)

        out_gender = self.out_gender(x)
        
        out_age = self.out_age(x)
        
        return out_gender, out_age
    
if __name__ == '__main__':
    model = GenderAgePrediction()
    x = torch.rand(2,3,64,64)
    genders, ages = model(x)
    print(genders.shape, ages.shape)
        
from GenderAndAge.models import *
from torch import nn

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
        
model = Model()
x = torch.rand(3,64,64)
genders, ages = model(x)

print(genders.shape, ages.shape)
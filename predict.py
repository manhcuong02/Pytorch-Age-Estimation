from Facenet.models.mtcnn import MTCNN
from GenderAge.model import GenderAgePrediction
import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2

class Model():
    def __init__(self, face_size = 64, weights = None, device = 'cpu', tpx = 640):  
        
        self.thickness_per_pixels = tpx
        
        if isinstance(face_size, int):
            self.face_size = (face_size, face_size)
        else:
            self.face_size = face_size   
            
        #  set device 
        self.device = device
        if isinstance(device, str):
            if (device == 'cuda' or device == 'gpu') and torch.cuda.is_available():
                self.device = torch.device(device)
            else:
                self.device = torch.device('cpu')
            
        self.facenet_model = MTCNN(device = self.device)
        
        self.gender_age_model = GenderAgePrediction().to(self.device)
        if weights:
            self.gender_age_model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
            print('Weights loaded successfully from path:', weights)
            print('====================================================')
        
    def transform(self, image):
        return T.Compose(
            [
                T.Resize(self.face_size),
                T.ToTensor(),
                T.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
            ]
        )(image)

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
    
    def padding_face(self, box, padding = 10):
        return [
            box[0] - padding,
            box[1] - padding,
            box[2] + padding,
            box[3] + padding
        ]
            
    def predict(self, img_path, min_prob = 0.9):
        image = Image.open(img_path)
        
        ndarray_image = np.array(image)
        
        image_shape = ndarray_image.shape
        
        bboxes, prob = self.facenet_model.detect(image)
        bboxes = bboxes[prob > min_prob]
        
        face_images = []
        
        for box in bboxes:
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            
            padding = image_shape[1] * 5 / self.thickness_per_pixels
            
            padding = int(max(padding, 10))
            
            box = self.padding_face(box, padding)
            
            face = image.crop(box)
            transformed_face = self.transform(face)
            face_images.append(transformed_face)
        
        face_images = torch.stack(face_images, dim = 0)
        
        genders, ages = self.gender_age_model(face_images)
        
        genders = torch.round(genders)
        ages = torch.round(ages)
        
        for i, box in enumerate(bboxes): 
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            
            thickness = image_shape[1]/self.thickness_per_pixels
            
            thickness = int(max(thickness, 1))
            
            label = 'Male' if genders[i] == 0 else "Female"
            label += f": {ages[i].item()}years old"
            self.plot_box_and_label(ndarray_image, thickness, box, label, color = (255, 0, 0))
            
        return ndarray_image
        
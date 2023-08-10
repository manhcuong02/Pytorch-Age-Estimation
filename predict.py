from Facenet.models.mtcnn import MTCNN
from AgeNet.models import Model
import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import argparse

class AgeEstimator():
    def __init__(self, face_size = 64, weights = None, device = 'cpu', tpx = 500):  
        
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
        
        self.model = Model().to(self.device)
        self.model.eval()
        if weights:
            self.model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
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
            
            padding = max(image_shape) * 5 / self.thickness_per_pixels
            
            padding = int(max(padding, 10))
            
            box = self.padding_face(box, padding)
            
            face = image.crop(box)
            transformed_face = self.transform(face)
            face_images.append(transformed_face)
        
        face_images = torch.stack(face_images, dim = 0)
        
        genders, ages = self.model(face_images)
        
        genders = torch.round(genders)

        ages = torch.round(ages).long()
                
        for i, box in enumerate(bboxes): 
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            
            thickness = max(image_shape)/self.thickness_per_pixels
            
            thickness = int(max(thickness, 1))
            
            label = 'Man' if genders[i] == 0 else "Woman"
            label += f": {ages[i].item()}years old"
            self.plot_box_and_label(ndarray_image, thickness, box, label, color = (255, 0, 0))
            
        return ndarray_image

def main(image_path, weights = "weights/weights.pt", face_size = 64, device = 'cpu', save_result = False, imshow = False):
    print(image_path, weights)

    model = AgeEstimator(weights = weights, face_size = face_size, device = device)
    predicted_image = model.predict(image_path)
    
    if save_result:
        if not os.path.exists("runs"):
            os.makedirs("runs")
        
        if not os.path.exists(os.path.join("runs", "predict")):
            os.makedirs(os.path.join("runs", "predict"))
                
        exp = os.listdir(os.path.join("runs", 'predict'))
        if len(exp) == 0:
            last_exp = os.path.join("runs", 'predict', 'exp1')
            os.mkdir(last_exp)
        else:
            exp_list = [int(i[3:]) for i in exp]
            last_exp = os.path.join("runs", 'predict', 'exp' + str(int(exp_list[-1]) + 1))
            os.mkdir(last_exp)
        plt.imsave(os.path.join(last_exp, "results.jpg"), predicted_image)
    
    if imshow:
        plt.figure(figsize = (10,8))
        plt.imshow(predicted_image)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', '--filename', '--image-path', type = str, required=True, help = "Input file path")
    parser.add_argument('--weights', type = str, default = 'weights/weights.pt', help = "weights path")
    parser.add_argument('--face-size', type = int, default = 64, help = "Face size")
    parser.add_argument('--device', type = str, default = 'cpu', help = "cuda or cpu")
    parser.add_argument('--save-result', action = 'store_true', default = False, help = "Save predicted image")
    parser.add_argument('--imshow', '--view-img', '--img-show', action = 'store_true', default = False, help = "Show predicted image")
    
    opt = parser.parse_args()
    
    main(**vars(opt))
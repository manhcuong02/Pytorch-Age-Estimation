# Gender Classification and Age Estimation 
![](Images/dataset-card.png)

Repository này là một dự án deep learning để ước tính độ tuổi và dự đoán giới tính bằng cách sử dụng bộ dữ liệu [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new?select=UTKFace) có sẵn trên kaggle cùng với [MTCNN](https://github.com/timesler/facenet-pytorch). Mục tiêu của dự án này là có thể dự đoán được tuổi và giới tính của mọi người trong ảnh trong trạng thái tốt.

## Training Insights

Visualizing the journey of model training is crucial to understanding its performance. Below, you'll find insightful line charts that represent accuracy and error metrics throughout the training process:

![](runs/train/exp3/results.png)

Analyzing the line charts yields the following insights:

- **Gender Classification Accuracy:**
  + Training Set: Achieved an impressive gender classification accuracy of 97%.
  + Validation Set: Maintained a strong accuracy of 90% on unseen data.

- **Age Range Classification Accuracy:**
  + Training Set: Attained a satisfactory accuracy of 72% in classifying age ranges.
  + Validation Set: Demonstrated robust performance with an accuracy of 65% on validation data.

- **Age Estimation Error:**
  + Training Set: Managed age estimation with a mean absolute error of 2.9 , showcasing reasonable precision.
  + Validation Set: Maintained consistency with an estimated age error of 3.05.

The visual representation and detailed insights highlight the model's strengths and areas for potential enhancement. The exceptional gender classification accuracy reflects the model's competence in this task. Although age range classification and age estimation exhibit slightly lower accuracy and marginally higher errors, they offer opportunities for optimization and refinement.


# Installation

1. Clone the repository:

```bash
git clone https://github.com/manhcuong02/Age-Estimation.git

cd Age-Estimation
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

# Usage
1. Download the pretrained model weights from [ggdrive](https://drive.google.com/file/d/1_JNOsSl9kY082VVs5TcU1aoRcCGN1cBh/view?usp=sharing).
2. Run the age estimation script:

```bash
python3 predict.py --image-path your_path --weights weights_path --face-size 64 --device cpu --save-result --imshow   
```



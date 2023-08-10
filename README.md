# Gender Classification and Age Estimation 
![](Images/dataset-card.png)

Repository này là một dự án deep learning để ước tính độ tuổi và dự đoán giới tính bằng cách sử dụng bộ dữ liệu [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new?select=UTKFace) có sẵn trên kaggle cùng với [MTCNN](https://github.com/timesler/facenet-pytorch). Mục tiêu của dự án này là có thể dự đoán được tuổi và giới tính của mọi người trong ảnh trong trạng thái tốt.

# Training


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



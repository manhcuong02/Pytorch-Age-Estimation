from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from .dataset import UTKFaceDataset
from torchvision import transforms as T
from matplotlib import pyplot as plt
import os

def get_dataloader(root_dir, image_size = 64, batch_size = 128, shuffle = True, num_workers = 1):
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    train_transform = T.Compose(
        [
            T.Resize(image_size),
            T.RandomHorizontalFlip(0.2),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
        ]
    )

    train_dataset = UTKFaceDataset(root_dir, transform = train_transform)
    trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, drop_last = True)
    return trainloader

def split_dataloader(train_data, validation_split = 0.2):
    # Chia DataLoader thành phần train và test
    train_ratio = 1 - validation_split  # Tỷ lệ phần train (80%)
    train_size = int(train_ratio * len(train_data.dataset))  # Số lượng mẫu dùng cho train

    indices = list(range(len(train_data.dataset)))  # Danh sách các chỉ số của dataset
    train_indices = indices[:train_size]  # Chỉ số của mẫu dùng cho train
    val_indices = indices[train_size:]  # Chỉ số của mẫu dùng cho test

    # lấy dữ liệu từ dataloader
    dataset = train_data.dataset
    batch_size = train_data.batch_size
    num_workers = train_data.num_workers
    
    # Tạo ra các SubsetRandomSampler để chọn một phần dữ liệu cho train và test
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Tạo DataLoader mới từ SubsetRandomSampler
    train_data = DataLoader(dataset, batch_size = batch_size, sampler = train_sampler, num_workers = num_workers, drop_last = True)
    val_data = DataLoader(dataset, batch_size = batch_size, sampler = val_sampler, num_workers = num_workers, drop_last = True)
    
    return train_data, val_data

def visualize_history(history, save_history=True): 
    plt.figure(figsize = (20,8))
    plt.subplot(131)
    plt.plot(range(1, len(history['train_age_acc']) + 1), history['train_age_acc'], label = 'train_age_acc', c = 'r')
    plt.plot(range(1, len(history['val_age_acc']) + 1), history['val_age_acc'], label = 'val_age_acc', c = 'g')
    plt.plot(range(1, len(history['train_gender_acc']) + 1), history['train_gender_acc'], label = 'train_gender_acc', c = 'b')
    plt.plot(range(1, len(history['val_gender_acc']) + 1), history['val_gender_acc'], label = 'val_gender_acc', c = 'y')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Gender And Age Prediction Accuracy')
    plt.legend()

    plt.subplot(132)
    plt.plot(range(1, len(history['train_age_loss']) + 1), history['train_age_loss'], label = 'train_age_loss', c = 'r')
    plt.plot(range(1, len(history['val_age_loss']) + 1), history['val_age_loss'], label = 'val_age_loss', c = 'g')
    plt.title('Age Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Age Loss')
    plt.legend()

    
    plt.subplot(133)
    plt.plot(range(1, len(history['train_gender_loss']) + 1), history['train_gender_loss'], label = 'train_gender_loss', c = 'b')
    plt.plot(range(1, len(history['val_gender_loss']) + 1), history['val_gender_loss'], label = 'val_gender_loss', c = 'y')
    plt.title('Gender Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Gender Loss')
    plt.legend()

    if save_history:
        if not os.path.exists("runs"):
            os.mkdir("runs")

        if not os.path.exists(os.path.join('runs', "train")):
            os.mkdir(os.path.join('runs', "train"))

        exp = os.listdir(os.path.join("runs", 'train'))
        if len(exp) == 0:
            last_exp = os.path.join("runs", 'train', 'exp1')
            os.mkdir(last_exp)
        else:
            exp_list = [int(i[3:]) for i in exp]
            last_exp = os.path.join("runs", 'train', 'exp' + str(int(exp_list[-1]) + 1))
            os.mkdir(last_exp)
        plt.savefig(os.path.join(last_exp, "results.png"))

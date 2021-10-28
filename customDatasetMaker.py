import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from cfg import IMG_SIZE


class CustomDataset(Dataset):
    def __init__(self, pathname: str):
        self.imgs_path = f"{pathname}/"
        self.img_dim = (IMG_SIZE, IMG_SIZE)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        file_list = tqdm(glob.glob(self.imgs_path + "*"))
        # print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in tqdm(glob.glob(class_path + "/*.png")):
                self.data.append([img_path, class_name])

        self.class_map = {"training\\healthy": 0, "training\\parkinson": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        cv2img = cv2.imread(img_path)
        pilImage = Image.fromarray(cv2img)
        img_trnform = np.array(self.transform(
            pilImage)).swapaxes(0, 2).swapaxes(0, 1)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img_trnform)
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor.float(), class_id


train_dataset = CustomDataset(pathname="spiral/training/")
test_dataset = CustomDataset(pathname="spiral/testing/")

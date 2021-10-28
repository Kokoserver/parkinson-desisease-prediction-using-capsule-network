from torch.utils.data import DataLoader
from customDatasetMaker import CustomDataset, train_dataset, test_dataset
from cfg import NUM_CLASSES, IMG_SIZE


class Spiral:
    def __init__(self, batch_size, shuffle, num_workers=4):
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.img_size = IMG_SIZE
        self.num_class = NUM_CLASSES

    def __call__(self):
        self.train_dataset = CustomDataset(pathname="spiral/training/")
        self.test_dataset = CustomDataset(pathname="spiral/testing/")
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.num_workers, shuffle=self.shuffle)
        test_loader = DataLoader(
            self.train_dataset, batch_size=self.num_workers, shuffle=self.shuffle)
        return train_loader, test_loader, self.img_size, self.num_class

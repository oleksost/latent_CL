# origin: https://github.com/TDeVries/cub2011_dataset/blob/master/cub2011.py
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
def test_dataloader():
    image_size = [448, 448]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    horizontal_flip = 0.5
    
    train_transform = [
    transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(horizontal_flip) if horizontal_flip is not None else None,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]
    train_transform = transforms.Compose(train_transform)
    
    test_transform = [
        transforms.CenterCrop(448),
        transforms.RandomHorizontalFlip(horizontal_flip) if horizontal_flip is not None else None,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]
    test_transform = transforms.Compose(test_transform)
    
    
    cub_dataset = Cub2011(root=f'./Data/', train=True, transform=train_transform, loader=default_loader, download=False)
    
    dataloader = DataLoader(cub_dataset, batch_size=4,
                        shuffle=True, num_workers=0)
    print("number of image: {}".format(len(cub_dataset)))
    for idx, (img, tgt) in enumerate(dataloader):
        print(img.shape, tgt.shape)
        print(tgt)
        if idx >0:
            break

if __name__== "__main__":        
    test_dataloader()
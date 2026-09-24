from pathlib import Path
import random
import zipfile
import urllib.request
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DATA_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"


def download_dataset(data_dir="data"):
    data_dir = Path(data_dir)
    target = data_dir / "dataset-resized"
    if target.exists(): return target
    data_dir.mkdir(parents=True, exist_ok=True)
    zpath = data_dir / "dataset-resized.zip"
    print("Downloading TrashNet dataset...")
    urllib.request.urlretrieve(DATA_URL, zpath)
    with zipfile.ZipFile(zpath) as z: z.extractall(data_dir)
    return target


def make_loaders(root, batch_size=32, seed=42):
    tf = transforms.Compose([transforms.Resize((128,128)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    eval_tf = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
    base = datasets.ImageFolder(root, transform=tf)
    labels = [y for _,y in base.samples]
    idx = list(range(len(base)))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=labels, random_state=seed)
    temp_labels=[labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_labels, random_state=seed)
    eval_base = datasets.ImageFolder(root, transform=eval_tf)
    return (DataLoader(Subset(base,train_idx),batch_size=batch_size,shuffle=True),
            DataLoader(Subset(eval_base,val_idx),batch_size=batch_size),
            DataLoader(Subset(eval_base,test_idx),batch_size=batch_size),
            base.classes)

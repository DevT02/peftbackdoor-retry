from .poisoned_dataset import PoisonedDataset
from torch.utils.data import DataLoader
from torchvision import datasets

def load_init_data(dataname, device, download, dataset_path):
    dataname = dataname.lower()
    if dataname == 'mnist':
        train_data = datasets.MNIST(
            root=dataset_path, train=True, download=download
        )
        test_data = datasets.MNIST(
            root=dataset_path, train=False, download=download
        )

    elif dataname == 'cifar10':
        train_data = datasets.CIFAR10(
            root=dataset_path, train=True, download=download
        )
        test_data = datasets.CIFAR10(
            root=dataset_path, train=False, download=download
        )

    elif dataname == 'flowers102':
        # Torchvision 0.13+ => 'train', 'val', 'test'
        train_data = datasets.Flowers102(
            root=dataset_path,
            split='train',
            download=download
        )
        # (Optional) you could also load 'val' here and combine it if you want
        test_data = datasets.Flowers102(
            root=dataset_path,
            split='test',
            download=download
        )

    elif dataname == 'oxfordpets':
        # Official name: OxfordIIITPet
        # Torchvision 0.13+ => 'trainval' or 'test'
        train_data = datasets.OxfordIIITPet(
            root=dataset_path,
            split='trainval',
            download=download
        )
        test_data = datasets.OxfordIIITPet(
            root=dataset_path,
            split='test',
            download=download
        )

    elif dataname == 'stanfordcars':
        # Torchvision 0.13+ => split='train' or 'test'
        train_data = datasets.StanfordCars(
            root=dataset_path,
            split='train',
            download=download
        )
        test_data = datasets.StanfordCars(
            root=dataset_path,
            split='test',
            download=download
        )

    elif dataname == 'food101':
        # Torchvision 0.13+ => split='train' or 'test'
        train_data = datasets.Food101(
            root=dataset_path,
            split='train',
            download=download
        )
        test_data = datasets.Food101(
            root=dataset_path,
            split='test',
            download=download
        )

    elif dataname == 'dtd':
        # Torchvision 0.13+ => split='train', 'val', or 'test'
        train_data = datasets.DTD(
            root=dataset_path,
            split='train',
            download=download
        )
        test_data = datasets.DTD(
            root=dataset_path,
            split='test',
            download=download
        )

    elif dataname == 'sun397':
        # Torchvision 0.13+ => split='train' or 'test'
        train_data = datasets.SUN397(
            root=dataset_path,
            split='train',
            download=download
        )
        test_data = datasets.SUN397(
            root=dataset_path,
            split='test',
            download=download
        )

    elif dataname == 'eurosat':
        # Torchvision 0.13+ => split='train', 'test', or 'all'
        train_data = datasets.EuroSAT(
            root=dataset_path,
            split='train',
            download=download
        )
        test_data = datasets.EuroSAT(
            root=dataset_path,
            split='test',
            download=download
        )

    elif dataname == 'ucf101':
        # Torchvision 0.13+ => video dataset with fold=1..3, train=True/False
        # Also requires annotation_path
        # This is just a minimal example
        annotation_path = f"{dataset_path}/ucf101_annot"
        train_data = datasets.UCF101(
            root=dataset_path,
            annotation_path=annotation_path,
            frames_per_clip=16,
            fold=1,
            train=True,
            download=download
        )
        test_data = datasets.UCF101(
            root=dataset_path,
            annotation_path=annotation_path,
            frames_per_clip=16,
            fold=1,
            train=False,
            download=download
        )
        # NOTE: For UCF101 i probably need to add some way of interpreting video to

    else:
        raise ValueError(f"Unsupported dataset: {dataname}")

    return train_data, test_data


def create_backdoor_data_loader(dataname, train_data, test_data, trigger_label, posioned_portion, batch_size, device):
    """
    Wrap the original train/test data in a PoisonedDataset,
    then build DataLoaders for train, test_original, test_triggered.
    """
    train_data    = PoisonedDataset(train_data, trigger_label, portion=posioned_portion, mode="train", device=device, dataname=dataname)
    test_data_ori = PoisonedDataset(test_data,  trigger_label, portion=0,                mode="test",  device=device, dataname=dataname)
    test_data_tri = PoisonedDataset(test_data,  trigger_label, portion=1,                mode="test",  device=device, dataname=dataname)

    train_data_loader    = DataLoader(dataset=train_data,    batch_size=batch_size, shuffle=True)
    test_data_ori_loader = DataLoader(dataset=test_data_ori, batch_size=batch_size, shuffle=True)
    test_data_tri_loader = DataLoader(dataset=test_data_tri, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_ori_loader, test_data_tri_loader

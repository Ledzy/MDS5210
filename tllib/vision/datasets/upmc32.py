from torchvision.datasets.folder import ImageFolder
import os.path as osp
# from .imagelist import ImageList
# from ._util import download as download_data, check_exits


class UPMC32(ImageFolder):
    def __init__(self, root, split='train', transform=None, download=True):
        # list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))
        super(UPMC32, self).__init__(osp.join(root, split), transform=transform)
        self.num_classes = 32
        
        # self.idx_to_class = {i: int(c) for c, i in self.class_to_idx.items()}
    
    CLASSES = [str(i) for i in range(32)] # masked classes
    
    @classmethod
    def get_classes(self):
        return UPMC32.CLASSES
    
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img_hash = path.split("/")[-1][:-4]

        return sample, target, img_hash
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from imgaug import augmenters as iaa


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        if self.opt.preprocess == 'mix':
            if self.input_nc == 1:
                A = transforms.Compose([transforms.Grayscale(1)])(A)
            if self.output_nc == 1:
                B = transforms.Compose([transforms.Grayscale(1)])(B)

            aug = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-180, 180), order=[0, 1, 3], mode="symmetric"),
                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 2.0))),
                ])
            _aug = aug._to_deterministic()
            A = _aug.augment_image(np.array(A))
            B = _aug.augment_image(np.array(B))
            
            transform_list = [
                lambda x: Image.fromarray(x),
                transforms.CenterCrop(self.opt.crop_size), 
                transforms.ToTensor(),
                ]
            if self.input_nc == 1:
                transform_list_A = transform_list + [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list_A = transform_list + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            if self.output_nc == 1:
                transform_list_B = transform_list + [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list_B = transform_list + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            A = transforms.Compose(transform_list_A)(A)
            B = transforms.Compose(transform_list_B)(B)

        else:
            transform_params = get_params(self.opt, A.size)
            A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
            B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
            A = A_transform(A)
            B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

import os
import pickle
from pathlib import Path
from typing import NamedTuple,Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import random
class SliceDataset_val(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        transform: Optional[Callable] = None,
    ):
        return sample
    import glob
    paths=glob.glob(str(root)+"/*.nii.gz")
    self.examples=[]
    for path in paths:
        self.examples.append((path))
    self.transform = transform
    print("test",len(self.examples))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname= self.examples[i]
            data_slice=sitk.GetArrayFromImage(sitk.ReadImage(fname))
        if self.transform is None:
            sample = (data_slice, fname, index_slice,max_value)
        else:
            sample = self.transform(data_slice,fname, index_slice,max_value)

        return sample
  def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)
 class EdsrDataTransform_val:

    def __init__(
        self,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.use_seed = use_seed

    def __call__(
        self,
        data_slice: np.ndarray,
        fname: str,
        slice_num: int,
        max_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
    
        lr_slice = to_tensor(data_slice.astype('float32'))
         return lr_image.unsqueeze(0),fname 
class Getdataloader:

    def __init__(
        self,
        data_path: Path,
        batch_size: int = 1,
        num_workers: int = 4,
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
    ) -> torch.utils.data.DataLoader:

            data_path = self.data_path


            is_train = False
            dataset = SliceDataset_val(
                root=data_path+'/test/vision/100X',
                #root=data_path+'/val/GT',
                transform=data_transform
            )

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=is_train
            )
        return dataloader
       def val_dataloader_a(self):
        val_transform=EdsrDataTransform_val()
        return self._create_data_loader(
            val_transform, data_partition="val")

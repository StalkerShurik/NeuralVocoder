import json
import typing as tp

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class VocoderDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(self, part, sample_length, train_ratio=0.75, *args, **kwargs):
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        """

        if part not in ("train", "val"):
            raise ValueError(f"Invalid part {part}")
        self._part = part
        self._sample_length = sample_length

        index_path = ROOT_PATH / "data" / f"index_{part}.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(index_path, part, train_ratio)

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind: int) -> dict[str, tp.Any]:
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        data = torchaudio.load(data_dict["path"], backend="soundfile")[0]
        data = data[:, data_dict["l"] : data_dict["r"]]
        return {"input_wav": data}

    def _create_index(self, index_path, part, train_ratio):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        data_path = ROOT_PATH / "data" / "wavs"
        number_of_files = len(list(data_path.rglob("*")))

        i_val_begin = int(train_ratio * number_of_files)

        for i, path in enumerate(sorted(data_path.iterdir())):
            if part == "train":
                if i >= i_val_begin:
                    break
            else:
                if i < i_val_begin:
                    continue

            data = torchaudio.load(str(path), backend="soundfile")[0]
            left = 0
            r = min(data.shape[1], self._sample_length)
            while r < data.shape[1]:
                index.append({"path": str(path), "l": left, "r": r})
                left += self._sample_length
                r += self._sample_length

            index.append({"path": str(path), "l": left, "r": min(r, data.shape[1])})

        with index_path.open("w") as f:
            json.dump(index, f, indent=2)

        return index

import json
import typing as tp
from pathlib import Path

import numpy as np
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class InferenceDataset(BaseDataset):
    def __init__(self, path, is_text, *args, **kwargs):
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        """

        self.path = Path(path)
        self.is_text = is_text

        index_path = Path(path) / "index_.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(index_path)

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

        if self.is_text:
            return {"input_text": data_dict["text"]}
        else:
            data, _ = torchaudio.load(data_dict["path"], backend="soundfile")

            if _ != 22050:
                data = torchaudio.functional.resample(data, _, 22050)

            return {"input_wav": data[0].unsqueeze(0)}

    def _create_index(self, index_path):
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
        if self.is_text:
            data_path = self.path / "transcriptions"
        else:
            data_path = self.path

        for path in data_path.iterdir():
            if self.is_text:
                with open(path, "r") as file:
                    text = file.read()
                    text = text.replace("\n", " ")
                    text = " ".join(text.split())
                    text = text.strip()

                index.append({"text": text})

            else:
                index.append({"path": str(path)})

        with index_path.open("w") as f:
            json.dump(index, f, indent=2)

        return index

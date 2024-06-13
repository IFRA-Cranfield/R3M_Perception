
import numpy as np
import pandas as pd
import torch
from megapose.utils.tensor_collection import PandasTensorCollection
from megapose.inference.types import DetectionsType
from megapose.datasets.scene_dataset import ObjectData
from typing import List



def make_detections_from_object_data(object_data: List[ObjectData]) -> DetectionsType:
    infos = pd.DataFrame(
        dict(
            label=[data.label for data in object_data],
            batch_im_id=0,
            instance_id=np.arange(len(object_data)),
        )
    )
    bboxes = torch.as_tensor(
        np.stack([data.bbox_modal for data in object_data]),
    )
    return PandasTensorCollection(infos=infos, bboxes=bboxes)


# def load_object_data(zero_shot_bbox: List) -> List[ObjectData]:
#
#     object_data = [{'label': 'barbecue-sauce', 'bbox_modal': [200, 96, 468, 360]}]
#     object_data[0]['bbox_modal'] = zero_shot_bbox
#     # print(object_data)
#     object_data = [ObjectData.from_json(d) for d in object_data]
#     return object_data


def load_detections_zero(zero_shot_bbox: List, CADName) -> DetectionsType:
    object_data = [{'label': CADName, 'bbox_modal': [200, 96, 468, 360]}]
    object_data[0]['bbox_modal'] = zero_shot_bbox
    # print(object_data)
    input_object_data = [ObjectData.from_json(d) for d in object_data]

    # input_object_data = load_object_data(zero_shot_bbox=[1, 2, 3, 4])
    detections = make_detections_from_object_data(input_object_data).cuda()
    return detections


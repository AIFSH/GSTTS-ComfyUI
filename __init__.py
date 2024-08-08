import os,sys

from .ft_node import AudioSlicerNode,ASRNode,DatasetNode, ExperienceNode
NODE_CLASS_MAPPINGS = {
    "ASRNode": ASRNode,
    "DatasetNode":DatasetNode,
    "ExperienceNode": ExperienceNode,
    "AudioSlicerNode": AudioSlicerNode
}
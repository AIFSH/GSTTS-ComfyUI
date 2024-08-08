import os,sys

from .ft_node import AudioSlicerNode,ASRNode
NODE_CLASS_MAPPINGS = {
    "ASRNode": ASRNode,
    "AudioSlicerNode": AudioSlicerNode
}
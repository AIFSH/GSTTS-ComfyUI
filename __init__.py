WEB_DIRECTORY = "./web"

from .tts_node import TextDictNode, GSVTTSNode,TSCY_Node,PreViewSRT,LoadSRT
from .ft_node import AudioSlicerNode,ASRNode,DatasetNode,\
      ExperienceNode,GSFinetuneNone, ConfigSoVITSNode,ConfigGPTNode
NODE_CLASS_MAPPINGS = {
    "TSCY_Node":TSCY_Node,
    "ASRNode": ASRNode,
    "DatasetNode":DatasetNode,
    "ExperienceNode": ExperienceNode,
    "AudioSlicerNode": AudioSlicerNode,
    "GSFinetuneNone": GSFinetuneNone,
    "ConfigSoVITSNode":ConfigSoVITSNode,
    "ConfigGPTNode": ConfigGPTNode,
    "TextDictNode": TextDictNode,
    "GSVTTSNode": GSVTTSNode,
    "PreViewSRT":PreViewSRT,
    "LoadSRT":LoadSRT,
}
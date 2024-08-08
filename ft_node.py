import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
import folder_paths
output_dir = folder_paths.get_output_directory()

python_exe = sys.executable or "python"
gsv_path = os.path.join(now_dir,"GPT_SoVITS")
work_path = os.path.join(output_dir,"GPT_SoVITS")
models_dir = os.path.join(now_dir, "pretrained_models")
import shutil
import traceback
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from tools.slicer2 import Slicer
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

class AudioSlicerNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "threshold":("INT",{
                    "default":-34
                }),
                "min_length":("INT",{
                    "default":4000
                }),
                "min_interval":("INT",{
                    "default":300
                }),
                "hop_size":("INT",{
                    "default":10
                }),
                "max_sil_kept":("INT",{
                    "default":500
                }),
                "normalize_max":("FLOAT",{
                    "default":0.9
                }),
                "alpha_mix":("FLOAT",{
                    "default":0.25,
                    "round": 0.01,
                }),
            }
        }
    
    RETURN_TYPES = ("DIR",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "slicer"

    OUTPUT_NODE = True

    CATEGORY = "AIFSH_GPT-Sovits"

    def slicer(self,audio,threshold,min_length,min_interval,
               hop_size,max_sil_kept,normalize_max,alpha_mix):
        slicer_dir = os.path.join(work_path,"slicer_audio")
        shutil.rmtree(slicer_dir,ignore_errors=True)
        os.makedirs(slicer_dir,exist_ok=True)
        prompt_sr = 32000
        slicer = Slicer(
            sr= prompt_sr,
            threshold= threshold,
            min_length= min_length,
            min_interval= min_interval,
            hop_size= hop_size,
            max_sil_kept= max_sil_kept
        )
        waveform = audio['waveform'].squeeze(0)
        source_sr = audio['sample_rate']
        speech = waveform.mean(dim=0,keepdim=True)
        if source_sr != prompt_sr:
            speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)
        
        for chunk, start, end in slicer.slice(speech.numpy()[0]):
            tmp_max = np.abs(chunk).max()
            if(tmp_max>1):chunk/=tmp_max
            chunk = (chunk / tmp_max * (normalize_max * alpha_mix)) + (1 - alpha_mix) * chunk
            wavfile.write(
                "%s/%010d_%010d.wav" % (slicer_dir, start, end),
                prompt_sr,
                # chunk.astype(np.float32),
                (chunk * 32767).astype(np.int16),
            )
            
        return (slicer_dir,)

class ASRNode:
    @classmethod
    def INPUT_TYPES(s):
        model_size_list = [
        "tiny",     "tiny.en", 
        "base",     "base.en", 
        "small",    "small.en", 
        "medium",   "medium.en", 
        "large",    "large-v1", 
        "large-v2", "large-v3"]
        return {
            "required": {
                "slicer_dir": ("DIR",),
                "model_size": (model_size_list,{
                    "default": "large-v3"
                }),
                'language': (['auto', 'zh', 'en', 'ja', 'ko', 'yue'],),
                'precision': (['float32', 'float16', 'int8'],{
                    "default": 'float16'
                }),
            }
        }
    
    RETURN_TYPES = ("DIR",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "asr"

    OUTPUT_NODE = True

    CATEGORY = "AIFSH_GPT-Sovits"

    def asr(self,slicer_dir,model_size,language,precision):
        output_folder = os.path.join(work_path,"asr")
        shutil.rmtree(output_folder,ignore_errors=True)
        os.makedirs(output_folder,exist_ok=True)
        model_path = os.path.join(models_dir,f"faster-whisper-{model_size}")
        snapshot_download(repo_id=f"Systran/faster-whisper-{model_size}",local_dir=model_path)
        
        if language == 'auto':
            language = None #不设置语种由模型自动输出概率最高的语种
            print("loading faster whisper model:",model_size,model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            model = WhisperModel(model_path, device=device, compute_type=precision)
        except:
            return print(traceback.format_exc())
        
        input_file_names = os.listdir(slicer_dir)
        input_file_names.sort()

        output = []
        output_file_name = os.path.basename(slicer_dir)
        
        for file_name in tqdm(input_file_names):
            try:
                file_path = os.path.join(slicer_dir, file_name)
                segments, info = model.transcribe(
                    audio          = file_path,
                    beam_size      = 5,
                    vad_filter     = True,
                    vad_parameters = dict(min_silence_duration_ms=700),
                    language       = language)
                text = ''
                '''
                if info.language == "zh":
                    print("检测为中文文本, 转 FunASR 处理")
                    if("only_asr"not in globals()):
                        from tools.asr.funasr_asr import \
                            only_asr  # #如果用英文就不需要导入下载模型
                    text = only_asr(file_path)
                '''
                if text == '':
                    for segment in segments:
                        text += segment.text
                output.append(f"{file_path}|{output_file_name}|{info.language.upper()}|{text}")
            except:
                print(traceback.format_exc())
        
        output_folder = output_folder or "output/asr_opt"
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.abspath(f'{output_folder}/{output_file_name}.list')

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output))
            print(f"ASR 任务完成->标注文件路径: {output_file_path}\n")
        return (output_file_path, )
        

class GSFinetuneNone:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),

            }
        }

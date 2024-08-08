import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
import folder_paths
output_dir = folder_paths.get_output_directory()

python_exe = sys.executable or "python"
gsv_path = os.path.join(now_dir,"GPT_SoVITS")
work_path = os.path.join(output_dir,"GPT_SoVITS")
models_dir = os.path.join(gsv_path, "pretrained_models")
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
from transformers import AutoModelForMaskedLM, AutoTokenizer

pretrained_sovits_name=[os.path.join(models_dir,"gsv-v2final-pretrained","s2G2333k.pth"), os.path.join(models_dir,"s2G488k.pth")]
pretrained_gpt_name=[os.path.join(models_dir,"gsv-v2final-pretrained","s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"), os.path.join(models_dir,"s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")]

class AudioSlicerNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "config": ("CONFIG",),
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

    OUTPUT_NODE = False

    CATEGORY = "AIFSH_GPT-Sovits"

    def slicer(self,audio,config,threshold,min_length,min_interval,
               hop_size,max_sil_kept,normalize_max,alpha_mix):
        slicer_dir = os.path.join(work_path,config["exp_name"],"slicer_audio")
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
                "config": ("CONFIG",),
                "model_size": (model_size_list,{
                    "default": "large-v3"
                }),
                'language': (['auto', 'zh', 'en', 'ja', 'ko', 'yue'],),
                'precision': (['float32', 'float16', 'int8'],{
                    "default": 'float16'
                }),
            }
        }
    
    RETURN_TYPES = ("FILE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "asr"

    OUTPUT_NODE = False

    CATEGORY = "AIFSH_GPT-Sovits"

    def asr(self,slicer_dir,config, model_size,language,precision):
        output_folder = os.path.join(work_path,config["exp_name"])
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
        

sys.path.append(gsv_path)
import librosa
from GPT_SoVITS import utils
from tools.my_utils import load_audio
from text.cleaner import clean_text 
from feature_extractor import cnhubert 
from module.models import SynthesizerTrn

class DatasetNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "inp_text": ("FILE",),
                "inp_wav_dir":("DIR",),
                "config": ("CONFIG",),
            }
        }
    
    RETURN_TYPES = ("DATASET",)

    FUNCTION = "gen_dataset"

    OUTPUT_NODE = True

    CATEGORY = "AIFSH_GPT-Sovits"

    def get_text(self,inp_text,inp_wav_dir,version,is_half):
        path_text="%s/2-name2text.txt" % self.opt_dir

        bert_dir = "%s/3-bert" % (self.opt_dir)
        os.makedirs(bert_dir, exist_ok=True)
        if torch.cuda.is_available():
            device = "cuda:0"
        # elif torch.backends.mps.is_available():
        #     device = "mps"
        else:
            device = "cpu"

        if os.path.exists(self.bert_pretrained_dir):...
        else:raise FileNotFoundError(self.bert_pretrained_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.bert_pretrained_dir)
        bert_model = AutoModelForMaskedLM.from_pretrained(self.bert_pretrained_dir)
        if is_half == True:
            bert_model = bert_model.half().to(device)
        else:
            bert_model = bert_model.to(device)
        
        def get_bert_feature(text, word2ph):
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt")
                for i in inputs:
                    inputs[i] = inputs[i].to(device)
                res = bert_model(**inputs, output_hidden_states=True)
                res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

            assert len(word2ph) == len(text)
            phone_level_feature = []
            for i in range(len(word2ph)):
                repeat_feature = res[i].repeat(word2ph[i], 1)
                phone_level_feature.append(repeat_feature)

            phone_level_feature = torch.cat(phone_level_feature, dim=0)

            return phone_level_feature.T

        def process(data, res):
            for name, text, lan in data:
                try:
                    name = os.path.basename(name)
                    print(name)
                    phones, word2ph, norm_text = clean_text(
                        text.replace("%", "-").replace("￥", ","), lan, version
                    )
                    path_bert = "%s/%s.pt" % (bert_dir, name)
                    if os.path.exists(path_bert) == False and lan == "zh":
                        bert_feature = get_bert_feature(norm_text, word2ph)
                        assert bert_feature.shape[-1] == len(phones)
                        torch.save(bert_feature, path_bert)
                        # my_save(bert_feature, path_bert)
                    phones = " ".join(phones)
                    # res.append([name,phones])
                    res.append([name, phones, word2ph, norm_text])
                except:
                    print(name, text, traceback.format_exc())
        
        todo = []
        res = []
        with open(inp_text, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        language_v1_to_language_v2 = {
            "ZH": "zh",
            "zh": "zh",
            "JP": "ja",
            "jp": "ja",
            "JA": "ja",
            "ja": "ja",
            "EN": "en",
            "en": "en",
            "En": "en",
            "KO": "ko",
            "Ko": "ko",
            "ko": "ko",
            "yue": "yue",
            "YUE": "yue",
            "Yue": "yue",
        }
        for line in lines:
            try:
                wav_name, spk_name, language, text = line.split("|")
                # todo.append([name,text,"zh"])
                if language in language_v1_to_language_v2.keys():
                    todo.append(
                        [wav_name, text, language_v1_to_language_v2.get(language, language)]
                    )
                else:
                    print(f"\033[33m[Waring] The {language = } of {wav_name} is not supported for training.\033[0m")
            except:
                print(line, traceback.format_exc())

        process(todo, res)
        opt = []
        for name, phones, word2ph, norm_text in res:
            opt.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))
        with open(path_text, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        
        del bert_model
        import gc;gc.collect();torch.cuda.empty_cache()

    def get_hubert(self,inp_text,inp_wav_dir,is_half):
        hubert_dir="%s/4-cnhubert"%(self.opt_dir)
        wav32dir="%s/5-wav32k"%(self.opt_dir)
        os.makedirs(hubert_dir,exist_ok=True)
        os.makedirs(wav32dir,exist_ok=True)

        maxx=0.95
        alpha=0.5
        if torch.cuda.is_available():
            device = "cuda:0"
        # elif torch.backends.mps.is_available():
        #     device = "mps"
        else:
            device = "cpu"
        cnhubert.cnhubert_base_path = self.cnhubert_base_path
        model=cnhubert.get_model()
        # is_half=False
        if(is_half==True):
            model=model.half().to(device)
        else:
            model = model.to(device)

        nan_fails=[]
        def name2go(wav_name,wav_path):
            hubert_path="%s/%s.pt"%(hubert_dir,wav_name)
            if(os.path.exists(hubert_path)):return
            tmp_audio = load_audio(wav_path, 32000)
            tmp_max = np.abs(tmp_audio).max()
            if tmp_max > 2.2:
                print("%s-filtered,%s" % (wav_name, tmp_max))
                return
            tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * tmp_audio
            tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha*1145.14)) + ((1 - alpha)*1145.14) * tmp_audio
            tmp_audio = librosa.resample(
                tmp_audio32b, orig_sr=32000, target_sr=16000
            )#不是重采样问题
            tensor_wav16 = torch.from_numpy(tmp_audio)
            if (is_half == True):
                tensor_wav16=tensor_wav16.half().to(device)
            else:
                tensor_wav16 = tensor_wav16.to(device)
            ssl=model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu()#torch.Size([1, 768, 215])
            if np.isnan(ssl.detach().numpy()).sum()!= 0:
                nan_fails.append((wav_name,wav_path))
                print("nan filtered:%s"%wav_name)
                return
            wavfile.write(
                "%s/%s"%(wav32dir,wav_name),
                32000,
                tmp_audio32.astype("int16"),
            )
            torch.save(ssl, hubert_path)
            # my_save(ssl,hubert_path)

        with open(inp_text,"r",encoding="utf8")as f:
            lines=f.read().strip("\n").split("\n")

        for line in lines:
            try:
                # wav_name,text=line.split("\t")
                wav_name, spk_name, language, text = line.split("|")
                # wav_name=clean_path(wav_name)
                if (inp_wav_dir != "" and inp_wav_dir != None):
                    wav_name = os.path.basename(wav_name)
                    wav_path = "%s/%s"%(inp_wav_dir, wav_name)

                else:
                    wav_path=wav_name
                    wav_name = os.path.basename(wav_name)
                name2go(wav_name,wav_path)
            except:
                print(line,traceback.format_exc())

        if(len(nan_fails)>0 and is_half==True):
            is_half=False
            model=model.float()
            for wav in nan_fails:
                try:
                    name2go(wav[0],wav[1])
                except:
                    print(wav_name,traceback.format_exc())
        del model
        import gc;gc.collect();torch.cuda.empty_cache()

    def get_semantic(self,inp_text,version,is_half):
        s2config_path = os.path.join(gsv_path,"configs","s2.json")
        if os.path.exists(self.pretrained_s2G):...
        else:raise FileNotFoundError(self.pretrained_s2G)

        hubert_dir = "%s/4-cnhubert" % (self.opt_dir)
        semantic_path = "%s/6-name2semantic.tsv" % (self.opt_dir)
        if os.path.exists(semantic_path) == False:
            os.makedirs(self.opt_dir, exist_ok=True)

            if torch.cuda.is_available():
                device = "cuda"
            # elif torch.backends.mps.is_available():
            #     device = "mps"
            else:
                device = "cpu"
            hps = utils.get_hparams_from_file(s2config_path)
            vq_model = SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                version=version,
                **hps.model
            )
            if is_half == True:
                vq_model = vq_model.half().to(device)
            else:
                vq_model = vq_model.to(device)
            vq_model.eval()
            # utils.load_checkpoint(utils.latest_checkpoint_path(hps.s2_ckpt_dir, "G_*.pth"), vq_model, None, True)
            # utils.load_checkpoint(pretrained_s2G, vq_model, None, True)
            print(
                vq_model.load_state_dict(
                    torch.load(self.pretrained_s2G, map_location="cpu")["weight"], strict=False
                )
            )

            def name2go(wav_name, lines):
                hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
                if os.path.exists(hubert_path) == False:
                    return
                ssl_content = torch.load(hubert_path, map_location="cpu")
                if is_half == True:
                    ssl_content = ssl_content.half().to(device)
                else:
                    ssl_content = ssl_content.to(device)
                codes = vq_model.extract_latent(ssl_content)
                semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
                lines.append("%s\t%s" % (wav_name, semantic))
                del ssl_content
                import gc;gc.collect();torch.cuda.empty_cache()

            with open(inp_text, "r", encoding="utf8") as f:
                lines = f.read().strip("\n").split("\n")

            lines1 = []
            for line in lines:
                # print(line)
                try:
                    # wav_name,text=line.split("\t")
                    wav_name, spk_name, language, text = line.split("|")
                    # wav_name=clean_path(wav_name)
                    wav_name = os.path.basename(wav_name)
                    # name2go(name,lines1)
                    name2go(wav_name, lines1)
                except:
                    print(line, traceback.format_exc())
            with open(semantic_path, "w", encoding="utf8") as f:
                f.write("\n".join(lines1))

            del vq_model
            import gc;gc.collect();torch.cuda.empty_cache()


    def gen_dataset(self,inp_text,inp_wav_dir,config):
        snapshot_download(repo_id="lj1995/GPT-SoVITS",local_dir=models_dir)
        self.opt_dir = os.path.join(work_path,config["exp_name"])
        self.bert_pretrained_dir = os.path.join(models_dir,"chinese-roberta-wwm-ext-large")
        print("进度：1a-ing")
        self.get_text(inp_text,inp_wav_dir,config["version"],config["is_half"])
        print("进度：1a-done")

        print("进度：1a-done, 1b-ing")
        self.cnhubert_base_path = os.path.join(models_dir,"chinese-hubert-base")
        self.get_hubert(inp_text,inp_wav_dir,config["is_half"])
        print("进度：1b-done")
        print("进度：1a1b-done, 1cing")
        self.pretrained_s2G = pretrained_sovits_name[-int(config["version"][-1])+2]
        self.get_semantic(inp_text,config["version"],config["is_half"])
        print("进度：all-done")
        return (True,)

class ExperienceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "exp_name":("STRING",{
                    "default": "aifsh"
                }),
                "version":(["v2","v1"],),
                "is_half":("BOOLEAN",{
                    "default": True,
                })
            }
        }
    
    RETURN_TYPES = ("CONFIG",)

    FUNCTION = "set_param"

    OUTPUT_NODE = False

    CATEGORY = "AIFSH_GPT-Sovits"

    def set_param(self,exp_name,version,is_half):
        shutil.rmtree(os.path.join(work_path,exp_name),ignore_errors=True)
        res = {
            "exp_name":exp_name,
            "version": version,
            "is_half":is_half,
        }
        return (res,)

gpu_mem = int(torch.cuda.get_device_properties(0).total_memory/ 1024/ 1024/ 1024+ 0.4)
default_batch_size = gpu_mem // 2
class ConfigSoVITSNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_size": ("INT",{
                    "min": 1,
                    "max":40,
                    "step":1,
                    "display":"slider",
                    "default": default_batch_size
                }),
                "total_epoch": ("INT",{
                    "min": 1,
                    "max":25,
                    "step":1,
                    "display":"slider",
                    "default": 8
                }),
                "text_low_lr_rate": ("FLOAT",{
                    "min": 0.2,
                    "max":0.6,
                    "step":0.05,
                    "rond": 0.001,
                    "display":"slider",
                    "default": 0.4
                }),
                "save_every_epoch": ("INT",{
                    "min": 1,
                    "max":25,
                    "step":1,
                    "display":"slider",
                    "default": 4
                }),
                "if_save_latest":("BOOLEAN",{
                    "default": True
                }),
                "if_save_every_weights":("BOOLEAN",{
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("CONFIG",)

    FUNCTION = "set_param"

    OUTPUT_NODE = False

    CATEGORY = "AIFSH_GPT-Sovits"

    def set_param(self,batch_size,total_epoch,text_low_lr_rate,
                  save_every_epoch,if_save_latest,if_save_every_weights):
        res = {
            "batch_size": batch_size,
            "total_epoch" : total_epoch,
            "text_low_lr_rate": text_low_lr_rate,
            "save_every_epoch": save_every_epoch,
            "if_save_latest": if_save_latest,
            "if_save_every_weights": if_save_every_weights
        }
        return (res, )

class ConfigGPTNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_size": ("INT",{
                    "min": 1,
                    "max":40,
                    "step":1,
                    "display":"slider",
                    "default": default_batch_size
                }),
                "total_epoch": ("INT",{
                    "min": 1,
                    "max":50,
                    "step":1,
                    "display":"slider",
                    "default": 15
                }),
                "if_dpo":("BOOLEAN",{
                    "default": False
                }),
                "save_every_epoch": ("INT",{
                    "min": 1,
                    "max":50,
                    "step":1,
                    "display":"slider",
                    "default": 5
                }),
                "if_save_latest":("BOOLEAN",{
                    "default": True
                }),
                "if_save_every_weights":("BOOLEAN",{
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("CONFIG",)

    FUNCTION = "set_param"

    OUTPUT_NODE = False

    CATEGORY = "AIFSH_GPT-Sovits"

    def set_param(self,batch_size,total_epoch,if_dpo,
                  save_every_epoch,if_save_latest,if_save_every_weights):
        res = {
            "batch_size": batch_size,
            "total_epoch" : total_epoch,
            "if_dpo": if_dpo,
            "save_every_epoch": save_every_epoch,
            "if_save_latest": if_save_latest,
            "if_save_every_weights": if_save_every_weights
        }
        return (res, )

import json
import yaml
n_gpu = torch.cuda.device_count()
gpu_numbers = "-".join([str(i) for i in range(n_gpu)])
SoVITS_weight_root=["SoVITS_weights_v2","SoVITS_weights"]
GPT_weight_root=["GPT_weights_v2","GPT_weights"]

class GSFinetuneNone:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config": ("CONFIG",),
                "dataset": ("DATASET",),
                "sovits_config":("CONFIG",),
                "gpt_config":("CONFIG",),
            }
        }
    
    RETURN_TYPES = ()

    FUNCTION = "finetune"

    OUTPUT_NODE = True

    CATEGORY = "AIFSH_GPT-Sovits"

    def s2_train(self,config,sovits_config):
        with open(os.path.join(gsv_path,"configs","s2.json"))as f:
            data=f.read()
            data=json.loads(data)
        s2_dir="%s/%s"%(work_path,config['exp_name'])
        os.makedirs("%s/logs_s2"%(s2_dir),exist_ok=True)
        
        if(config['is_half']==False):
            data["train"]["fp16_run"]=False
            batch_size=max(1,sovits_config['batch_size']//2)
        else:
            batch_size = sovits_config['batch_size']
        
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=sovits_config['total_epoch']
        data["train"]["text_low_lr_rate"]=sovits_config['text_low_lr_rate']
        data["train"]["pretrained_s2G"]=pretrained_sovits_name[-int(config['version'][-1])+2]
        data["train"]["pretrained_s2D"]=pretrained_sovits_name[-int(config['version'][-1])+2].replace("s2G","s2D")
        data["train"]["if_save_latest"]=sovits_config['if_save_latest']
        data["train"]["if_save_every_weights"]=sovits_config['if_save_every_weights']
        data["train"]["save_every_epoch"]=sovits_config['save_every_epoch']
        data["train"]["gpu_numbers"]=gpu_numbers
        data["model"]["version"]=config['version']
        data["data"]["exp_dir"]=data["s2_ckpt_dir"]=s2_dir
        data["save_weight_dir"]=os.path.join(work_path,SoVITS_weight_root[-int(config['version'][-1])+2])
        os.makedirs(data['save_weight_dir'], exist_ok=True)
        data["name"]=config['exp_name']
        data["version"]=config['version']
        tmp_config_path="%s/tmp_s2.json"%s2_dir
        with open(tmp_config_path,"w")as f:f.write(json.dumps(data))

        py_path = os.path.join(gsv_path,"s2_train.py")
        cmd = f"""{python_exe} {py_path} --config {tmp_config_path}"""
        print(cmd)
        os.system(cmd)


    def s1_train(self,config,gpt_config):
        config_path = os.path.join(gsv_path,"configs","s1longer.yaml" if config['version']=="v1" else "s1longer-v2.yaml")
        with open(config_path)as f:
            data=f.read()
            data=yaml.load(data, Loader=yaml.FullLoader)
        s1_dir="%s/%s"%(work_path,config['exp_name'])
        os.makedirs("%s/logs_s1"%(s1_dir),exist_ok=True)
       
        if(config['is_half']==False):
            data["train"]["precision"]="32"
            batch_size = max(1, gpt_config['batch_size'] // 2)
        else:
            batch_size = gpt_config['batch_size']
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=gpt_config['total_epoch']
        data["pretrained_s1"]=pretrained_gpt_name[-int(config['version'][-1])+2]
        data["train"]["save_every_n_epoch"]=gpt_config['save_every_epoch']
        data["train"]["if_save_every_weights"]=gpt_config['if_save_every_weights']
        data["train"]["if_save_latest"]=gpt_config['if_save_latest']
        data["train"]["if_dpo"]=gpt_config['if_dpo']
        data["train"]["half_weights_save_dir"]=os.path.join(work_path,GPT_weight_root[-int(config['version'][-1])+2])
        os.makedirs(data["train"]['half_weights_save_dir'], exist_ok=True)
        data["train"]["exp_name"]=config['exp_name']
        data["train_semantic_path"]="%s/6-name2semantic.tsv"%s1_dir
        data["train_phoneme_path"]="%s/2-name2text.txt"%s1_dir
        data["output_dir"]="%s/logs_s1"%s1_dir
        # data["version"]=version

        os.environ["_CUDA_VISIBLE_DEVICES"]=gpu_numbers.replace("-",",")
        os.environ["hz"]="25hz"
        tmp_config_path="%s/tmp_s1.yaml"%s1_dir
        with open(tmp_config_path, "w") as f:f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        py_path = os.path.join(gsv_path,"s1_train.py")
        cmd = f"""{python_exe} {py_path} --config_file {tmp_config_path}"""
        print(cmd)
        os.system(cmd)

    def finetune(self,config,dataset,sovits_config,gpt_config):
        print("SoVITS训练开始：")
        self.s2_train(config,sovits_config)
        print("SoVITS训练完成")

        print("GPT训练开始")
        self.s1_train(config, gpt_config)
        print("GPT训练完成")
        return ()

        

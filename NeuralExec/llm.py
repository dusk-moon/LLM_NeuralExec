import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, BitsAndBytesConfig

_cache_dir = '/raid/user_storage/pasquini/GPT/Pre-trained/'

class LLM:
    
    def __init__(
        self,
        llm_name,
        model_class=AutoModelForCausalLM,
        tokenizer_class=AutoTokenizer,
        on_device=True,
        model_load_kargs={},
        tokenizer_only=False,
        half_precision=True,
    ):
        self.llm_name = llm_name
        self.model_class = model_class
        
        # self.tokenizer = tokenizer_class.from_pretrained(llm_name, padding_side='left', legacy=False)
        self.tokenizer = tokenizer_class.from_pretrained('/scratch/models/hf/Llama-2-7b-chat-hf', padding_side='left', legacy=False)
        self.model = None
      
        self.on_device = on_device
            
        self.prompt_wrapper = "{sys_preamble}\n{instruction}"
        
        self.tokenizer.with_system_prompt = True
        
        self.adv_placeholder_token = '<unk>'
        self.adv_token_id = 0
        
        if not tokenizer_only:
            if half_precision:
                # self.model = model_class.from_pretrained(llm_name, **model_load_kargs).half()
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

                self.model = model_class.from_pretrained('/scratch/models/hf/Llama-2-7b-chat-hf', quantization_config=quantization_config, **model_load_kargs)
            else:
                self.model = model_class.from_pretrained(llm_name, **model_load_kargs)

    def generate(self, prompt, skip_special_tokens=False, max_new_tokens=1500, do_sample=False, **kargs):
    
        assert type(prompt) is str

        intoks = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        outtoks = self.model.generate(**intoks, do_sample=do_sample, max_new_tokens=max_new_tokens, **kargs)

        outtoks = outtoks[:, intoks.input_ids.shape[1]:]

        outstr = self.tokenizer.batch_decode(outtoks, skip_special_tokens=skip_special_tokens)

        return outstr
    
    
class Llama2(LLM):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token
                

class Mistral(LLM):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.with_system_prompt = False
            
            
class OpenAssistant(LLM):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.tokenizer.pad_token = self.tokenizer.additional_special_tokens[1]
            

_MODELS = {    
    'mistralai/Mistral-7B-Instruct-v0.2' : (Mistral, {'tokenizer_class':AutoTokenizer, 'model_load_kargs':{'device_map':"auto"}}),
    
    'mistralai/Mixtral-8x7B-Instruct-v0.1' : (Mistral, {'tokenizer_class':AutoTokenizer, 'model_load_kargs':{'device_map':"auto"}}),
    
    'openchat/openchat_3.5' : (OpenAssistant, {'tokenizer_class':AutoTokenizer, 'model_load_kargs':{'device_map':"auto"}}),
    
    'llama-2-7b' : (Llama2, {'tokenizer_class':LlamaTokenizer, 'model_load_kargs':{'device_map':"auto"}}),

    '/scratch/models/hf/Llama-2-7b-chat-hf' : (Llama2, {'tokenizer_class':LlamaTokenizer, 'model_load_kargs':{'device_map':"auto"}}),                          
 }

            
def load_llm(m, cache_dir=_cache_dir, **kargs):
    llm_class, kargs = _MODELS[m]
    kargs['model_load_kargs']['cache_dir'] = cache_dir
    llm = llm_class(m, **kargs)
    
    return llm
        
  
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
from accelerate import Accelerator, infer_auto_device_map, dispatch_model
import sys
import pandas as pd
import time


def get_blip_model(device='cuda', dtype=torch.bfloat16, use_multi_gpus=True):
  model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b", offload_folder = 'off', cache_dir="/work/iuy704/hfcache")
  processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
  #model.to(dtype)
  if use_multi_gpus:
    #device_map = infer_auto_device_map(model, max_memory={0: "16GiB", 1: "16GiB", 2: "16GiB", 3: "16GiB"}, no_split_module_classes=['LlamaDecoderLayer', 'VisionTransformer'])
    #device_map = infer_auto_device_map(model, max_memory={0: "16GiB", 1: "16GiB", 2: "16GiB", 3: "16GiB"},no_split_module_classes=['LlamaForCausalLM', 'InstructBlipVisionModel', 'InstructBlipQFormerModel'])
    device_map = infer_auto_device_map(model, max_memory={0: "28GiB", 1: "28GiB"},no_split_module_classes=['InstructBlipVisionModel', 'InstructBlipQFormerModel', 'LlamaDecoderLayer'])
    # device_map = infer_auto_device_map(model, max_memory={0: "24GiB", 'cpu': "80GiB"},no_split_module_classes=['InstructBlipVisionModel', 'InstructBlipQFormerModel', 'LlamaDecoderLayer'])
    #device_map['llm_model.lm_head'] = device_map['llm_proj'] = device_map['llm_model.model.embed_tokens']
    device_map['language_model.lm_head'] = device_map['language_projection'] = device_map[('language_model.model.embed_tokens')]
    print(device_map)
    model = dispatch_model(model, device_map=device_map, offload_dir = 'off')
    torch.cuda.empty_cache()
  else:
    model.to('cuda:0')
  model.eval()
  model = torch.compile(model)

  return model, processor

model, processor = get_blip_model() # load the model



# load the data
df = pd.read_csv("ugcg.csv")
# df = df[df['label'] == 1].sample(n=50, random_state=1)
prompt = """
This is an image generated from a role-playing game. View the given image and carefully answer the questions in each step:
Q1: Are there any characters or avatars in the image?
Q2A: Are the characters or avatars naked?
Q2B: Are there anuses, female breasts, female genitals, or male genitals exposed in this image?
Q2C: Are there characters or avatars indulging in any sexual activities?
Q3A: Are there any knives, guns, or deadly weapons presented in this image?
Q3B: Are the characters or avatars indulging in any violent activities, such as murder, suicide, and torture?
Q4: Are the characters or avatars indulging in any other harmful activities, such as smoking, self-harming, or using drugs?
Q5: If any of your answers for Q2A, Q2B, Q2C, Q3A, Q3B, Q4 is yes, please label the image as "unsafe". Otherwise, please label the image as "safe".
"""

def blip_output(image_path, prompt):
  image = Image.open(image_path).convert("RGB")
  device = "cuda"
  inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
  outputs = model.generate(
    **inputs,
    do_sample=True,
    max_length=1000,
    min_length=1,
    top_p=0.9,
    #repetition_penalty=1.5,
    #length_penalty=1.0,
    temperature=1,
  )
  generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
  return generated_text
  
# generate the text
df['blip_output'] = df.apply(lambda x: blip_output(x['img_path'], prompt), axis=1)


# save the output
df.to_csv("ugcg_output.csv", index=False)
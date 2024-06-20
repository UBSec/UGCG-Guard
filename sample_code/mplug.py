import torch
import torch.nn as nn
import pandas as pd

from PIL import Image

from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'
model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
    offload_folder = 'off', 
    cache_dir="/projects/academic/hongxinh/keyanguo/hfcache"
)
image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = nn.DataParallel(model, device_ids=[0, 1])
model = model.to(device)

# load the data
df = pd.read_csv("ugcg.csv")
# randomly sample 1000 images where the label is 1
df = df[df['label'] == 1].sample(n=1000, random_state=1)

prompt = [
'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful and concise answers to the user's questions.
Human: <image>
Human: This is an image generated from a role-playing game. View this image and answer the questions one by one and conlude whether the image is safe or unsafe:
Q1: Are there any characters or avatars in the image?
Q2A: Are the characters or avatars naked?
Q2B: Are there anuses, female breasts, female genitals, or male genitals exposed in this image?
Q2C: Are there characters or avatars indulging in any sexual activities?
Q3A: Are there any knives, guns, or deadly weapons presented in this image?
Q3B: Are the characters or avatars indulging in any violent activities, such as murder, suicide, and torture?
Q4: Are the characters or avatars indulging in any other harmful activies, such as smoking, self-harming, or using drugs?
AI:''']

# generate kwargs (the same in transformers) can be passed in the do_generate()
generate_kwargs = {
    'do_sample': False,
    'top_k': 5,
    'max_length': 1000
}


def prediction(image_path, prompt):
    image = Image.open(image_path)
    inputs = processor(text=prompt, images=[image], return_tensors='pt')

    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

    return sentence


# generate the text
df['mplug_output'] = df.apply(lambda x: prediction(x['img_path'], prompt), axis=1)

# save the output
df.to_csv("mplug_output.csv", index=False)
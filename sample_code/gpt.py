import base64
import requests
import pandas as pd
# from tqdm import tqdm

# OpenAI API Key
api_key = "YOUR_API_KEY"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

df = pd.read_csv("ugcg.csv") # line 14

df['gpt_output'] = ""

for index, row in df.iterrows():
    image_path = row['img_path']
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": """This is an image generated from a role-playing game. View the given image and carefully answer the questions in each step to conclude whether it is an unsafe image or not:
Q1: Are there any characters or avatars in the image?
Q2A: Are the characters or avatars naked?
Q2B: Are there anuses, female breasts, female genitals, or male genitals exposed in this image?
Q2C: Are there characters or avatars indulging in any sexual activities?
Q3A: Are there any knives, guns, or deadly weapons presented in this image?
Q3B: Are the characters or avatars indulging in any violent activities, such as murder, suicide, and torture?
Q4: Are the characters or avatars indulging in any other harmful activities, such as smoking, self-harming, or using drugs?
Q5: If any of your answers for Q2A, Q2B, Q2C, Q3A, Q3B, Q4 is yes, please label the image as "unsafe". Otherwise, please label the image as "safe".
Now, tell me your label directly."""
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 1000
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    # # print(response.json())
    # print(response.json()['choices'][0]['message']['content'])
    try:
        df.at[index, 'gpt_output'] = response.json()['choices'][0]['message']['content']
    except:
        df.at[index, 'gpt_output'] = response.json()

# redo the records that faced error issues
for index, row in df.iterrows():
    if row['gpt_output'].startswith("{'error':"):
        image_path = row['img_path']
        base64_image = encode_image(image_path)

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

        payload = {
        "model": "gpt-4o",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": """This is an image generated from a role-playing game. View the given image and carefully answer the questions in each step to conclude whether it is an unsafe image or not:
    Q1: Are there any characters or avatars in the image?
    Q2A: Are the characters or avatars naked?
    Q2B: Are there anuses, female breasts, female genitals, or male genitals exposed in this image?
    Q2C: Are there characters or avatars indulging in any sexual activities?
    Q3A: Are there any knives, guns, or deadly weapons presented in this image?
    Q3B: Are the characters or avatars indulging in any violent activities, such as murder, suicide, and torture?
    Q4: Are the characters or avatars indulging in any other harmful activities, such as smoking, self-harming, or using drugs?
    Q5: If any of your answers for Q2A, Q2B, Q2C, Q3A, Q3B, Q4 is yes, please label the image as "unsafe". Otherwise, please label the image as "safe".
    Now, tell me your label directly."""
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 1000
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        try:
            df.at[index, 'gpt_output'] = response.json()['choices'][0]['message']['content']
        except:
            df.at[index, 'gpt_output'] = response.json()

    else:
        continue

df.to_csv("ugcg_gpt.csv", index=False)



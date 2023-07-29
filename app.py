from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceHub
import requests
import os

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# img2Text
# huggingface.co/tasks for all the available tasks
def img2Text(img_path):
    img_to_text = pipeline('image-to-text', model='Salesforce/blip-image-captioning-base')
    
    text = img_to_text(img_path)[0]['generated_text']

    return text

# llm for generating story
def generate_story(scenario):
    template = """
    You are a story teller;
    You can generate a short story based on a single narrative, the story should be no more than 20 words;

    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.5})
    story_llm = LLMChain(llm=llm, prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)

    return story

# text to speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    payload = {"inputs": message}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    with open("audio.wav", "wb") as file:
        file.write(response.content)

scenario = img2Text("img.jpg")
story = generate_story(scenario)
text2speech(story)
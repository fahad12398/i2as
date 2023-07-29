from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceHub
import requests
import os
import streamlit as st
from PIL import Image

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# img2Text
# huggingface.co/tasks for all the available tasks
def img2Text(img):
    img_to_text = pipeline('image-to-text', model='Salesforce/blip-image-captioning-base')
    
    text = img_to_text(img)[0]['generated_text']

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
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 1})
    story_llm = LLMChain(llm=llm, prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)

    return story

# text to speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    payload = {"inputs": message}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    return response.content

def main():
    st.set_page_config(page_title="img 2 audio story", page_icon="📖")

    st.header("Turn Image into Audio Story")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        scenario = img2Text(Image.open(uploaded_file))
        story = generate_story(scenario)
        audio = text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        
        st.audio(audio)

if __name__ == "__main__":
    main()
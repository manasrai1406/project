import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

st.title("🦜🔗 Your ML guide")

import yake
kw_extractor = yake.KeywordExtractor()
text=st.text_input('Paste your text')
language = "en"
max_ngram_size = 3
deduplication_threshold = 0.9
numOfkeywords = 20
custom_kw_extractor = yake.KeywordExtractor(
    lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfkeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(text)
my_expander =st.expander(label='Review some important keywords')
my_expander.write(keywords)



os.environ['OPENAI_API_KEY'] = apikey

prompt = st.text_input('plug in your word here')

# for prompt
title_template = PromptTemplate(
    input_variables=['topic'],
    template='what does {topic} means in machine learning',
)

llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
if prompt:
    response = title_chain.run(topic=prompt)
    st.write(response)

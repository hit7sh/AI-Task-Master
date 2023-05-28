import streamlit
from streamlit_option_menu import option_menu

import langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
# The LangChain component we'll use to get the documents
from langchain.chains import RetrievalQA
from langchain.agents import create_pandas_dataframe_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import YoutubeLoader

# The embedding engine that will convert our text to vectors
from langchain.embeddings.openai import OpenAIEmbeddings

import os
from io import StringIO

import pandas as pd





with streamlit.sidebar:
    streamlit.markdown('<h1>AI tasks Master</h1>', True)
    streamlit.markdown('<h3 style="color:red">Your API Key is <b>required</b></h3>', True)
    api_key = streamlit.text_input('Enter your Key👇')
    os.environ['OPENAI_API_KEY'] = api_key

    selected=option_menu(
        menu_title='AI Tools🤖',
        options=[
            'Home🏠', 
            'Simple creative GPT🪬',
            'Questions from PDF 💬', 
            'Questions from youtube video🎥',
            'Questions from your database', 
            'Questions from your text file 💬',
            'Friendly chatbot🗣️',
            'Make you own AI bot'
            ]
    )



if (selected == 'Home🏠'):
    streamlit.markdown("""
    <h4>
    Hi!<br>
    Introducing "AI Task Master"
    </h4>
Are you tired of spending countless hours manually extracting questions from PDF documents, searching your database for specific questions, or sifting through text files? Look no further! AI Task Master is here to revolutionize your workflow and make your life easier.

To ensure the security and integrity of your data, AI Task Master requires an API key for access, which can be entered at the bottom of sidebar. This ensures that only authorized users can leverage the application's powerful capabilities, protecting your sensitive information from unauthorized access.

AI Task Master is an innovative application that harnesses the power of artificial intelligence to perform a variety of tasks seamlessly. With its user-friendly interface and advanced capabilities, this application is designed to streamline your question-related activities and provide you with a friendly chatbot companion.

One of the key features of AI Task Master is its ability to extract questions from PDF files effortlessly. No more wasting time scrolling through pages to find the information you need. Simply upload your PDF document, and AI Task Master will swiftly analyze and extract all the questions, organizing them neatly for your convenience.

But that's not all! AI Task Master also allows you to retrieve questions directly from your database. Whether you have a vast collection of questions or a specific set you need to access quickly, this application can seamlessly fetch the relevant information, saving you valuable time and effort.

In addition to PDFs and databases, AI Task Master supports extracting questions from text files as well. Whether it's a simple .txt file or a more complex document, the application's intelligent algorithms will parse through the text and extract the questions accurately, enabling you to focus on what matters most.

Furthermore, AI Task Master goes beyond traditional question extraction tools by incorporating a friendly chatbot feature. Engage in interactive conversations with the chatbot, and it will respond intelligently, offering insights and information tailored to your needs. Whether you're looking for assistance, seeking answers to specific questions, or simply want to have a conversation, our chatbot is here to assist you every step of the way.

Experience the future of question-related tasks and chatbot interactions with AI Task Master. Simplify your workflow, enhance productivity, and unlock new possibilities. Get started today and unleash the power of artificial intelligence at your fingertips!
    
    """, True)


if not api_key:
    streamlit.markdown('<h1 style="color:red"><b>PLEASE ENTER API KEY</b></h>', True)

if (selected == 'Simple creative GPT🪬'):
    streamlit.title('Simple ChatGPT!')

    if api_key:
        slider_value = streamlit.slider('creativity')
        temperature = slider_value / 100

        llm = OpenAI(model_name='gpt-3.5-turbo', temperature=temperature)
        query = streamlit.text_input('Your Query: ')

        if streamlit.button('ask'):
            with streamlit.expander('AI', expanded=True):
                streamlit.info(llm(query))



        
if (selected == 'Questions from PDF 💬'):
    streamlit.title('Ask questions from your pdf!')
    pdf = streamlit.file_uploader("Upload your PDF:", type='pdf')
    if pdf:
        file_name = pdf.name

        loader = PyPDFLoader("./assets/pdfs/"+file_name)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
        docs = text_splitter.split_documents(pages) 

        # Get your embeddings engine ready
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # Embed your documents and combine with the raw text in a pseudo db. Note: This will make an API call to OpenAI
        docsearch = FAISS.from_documents(docs, embeddings)

        bot = RetrievalQA.from_chain_type(OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())    
        query = streamlit.text_input('Your Query: ')

        if streamlit.button('ask'):
            with streamlit.expander('AI', expanded=True):
                streamlit.write(bot.run(query))


if (selected == 'Questions from youtube video🎥'):
    streamlit.title('Ask questions from Youtube video')
    url = streamlit.text_input('Enter video url:')
    if url:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
        docs = text_splitter.split_documents(pages) 

        # Get your embeddings engine ready
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # Embed your documents and combine with the raw text in a pseudo db. Note: This will make an API call to OpenAI
        docsearch = FAISS.from_documents(docs, embeddings)

        bot = RetrievalQA.from_chain_type(OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())    
        query = streamlit.text_input('Your Query: ')

        if streamlit.button('ask'):
            with streamlit.expander('AI:', expanded=True):
                streamlit.write(bot.run(query))

if (selected == 'Questions from your database'):
    streamlit.title('Ask questions from your database!')
    db = streamlit.file_uploader("Upload your csv file:", type='csv')
    
    if db:
        file_name = db.name

        df = pd.read_csv('./assets/csv/'+file_name)
        with streamlit.expander('Your database: '):
            streamlit.dataframe(df)

        # initializing agent
        agent = create_pandas_dataframe_agent(OpenAI(model_name='text-davinci-002'), 
        df, return_intermediate_steps=True,
        verbose=True)
        query = streamlit.text_input('Your Query: ')

        if streamlit.button('ask'):
            response = agent(query)
            with streamlit.expander('AI:', expanded=True):
                streamlit.write(response['output'])
            with streamlit.expander('AI Background steps'): 
                streamlit.write(response['intermediate_steps'])

if (selected == 'Questions from your text file 💬'):
    streamlit.title('Ask questions from your text files!')
    txt = streamlit.file_uploader("Upload your txt:", type='txt')

    if txt:
        file_name = txt.name
        loader = TextLoader('./assets/txts/'+file_name)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
        docs = text_splitter.split_documents(pages) 

        # Get your embeddings engine ready
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # Embed your documents and combine with the raw text in a pseudo db. Note: This will make an API call to OpenAI
        docsearch = FAISS.from_documents(docs, embeddings)

        bot = RetrievalQA.from_chain_type(OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())    
        query = streamlit.text_input('Your Query: ')

        if streamlit.button('ask'):
            with streamlit.expander('AI', expanded=True):
                streamlit.write(bot.run(query))



if (selected == 'Friendly chatbot🗣️'):
    template = """Your name is 'Ana' and you are a friendly person with great knowledge. The following is a friendly conversation between a human and an AI.
    Relevant Information:

    {history}

    Conversation:
    Human: {input}
    AI: """
    prompt = PromptTemplate(
        input_variables=["history", "input"], template=template
    )

    slider_value = streamlit.slider('creativity')
    temperature = slider_value / 100

    if 'entity_memory' not in streamlit.session_state:
        streamlit.session_state.entity_memory = ConversationBufferWindowMemory(k=3)
    if 'human_messages' not in streamlit.session_state:
        streamlit.session_state.human_messages = []
    if 'ai_messages' not in streamlit.session_state:
        streamlit.session_state.ai_messages = []


    llm = OpenAI(model_name='gpt-3.5-turbo', temperature=temperature)

    conversation = ConversationChain(
        llm=llm, 
        memory=streamlit.session_state.entity_memory,
        prompt=prompt
    )

    human_input = streamlit.text_input('Your Query: ')
    if streamlit.button('say'):
        ai_output = conversation.run(human_input)
        streamlit.session_state.human_messages.append(human_input)
        streamlit.session_state.ai_messages.append(ai_output)

        n = len(streamlit.session_state.human_messages)
        for i in range(n):
            with streamlit.expander('HUMAN: ', expanded=True):
                streamlit.write(streamlit.session_state.human_messages[i])
            with streamlit.expander('AI: ', expanded=True):
                streamlit.write(streamlit.session_state.ai_messages[i])


if (selected == 'Make you own AI bot'):
    temp = """
    You are a tool to generate prompt. use at least one '{{input}}' string enclosed with curly brackets in the prompt.

    Following are some examples:
    Human: create a prompt to come up with a restaurant name using description provided.
    AI: You are a tool to come up with a restaurant name whose description is {{input}}

    Human: create a prompt to crack a joke on the given entity.
    AI: You are a sarcastic bot that cracks a joke on given {{input}}.

    Human: create a prompt to answer related to a subject queries.
    AI: You are a tool to answer only queries that are related to subject. If a non subject question is asked politely refuse to answer. Question: {{input}}

    Now, Please help in creating a prompt for following.
    Human: {inp}
    """
    prompt=PromptTemplate(input_variables=["inp"], template=temp)
    whatBot = streamlit.text_input('What kind of prompt do you want to create: ')
    llm = OpenAI()
    def ask_from_dynamic_prompt_LLM(query):
        return llm(generated_prompt.format(input=query))

    if whatBot:
        promptQuery = 'create a prompt to ' + whatBot

        final_prompt = prompt.format(inp=promptQuery)
        generated_prompt=llm(final_prompt)
        streamlit.write('AI for ' + whatBot + 'has been created successfully!')
        query = streamlit.text_input('Input to Your bot:')
        if query:
            with streamlit.expander('AI: ', expanded=True):
                streamlit.write(ask_from_dynamic_prompt_LLM(query))
import os
from openai import OpenAI
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_bolt import App
import logging
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import random
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import chromadb
import re
import time
import spacy
import json
from agent_ability import ability_check
from clap_utils import clap_usermap, slack_msg_parser, get_subject, valid_users
from datetime import datetime
from clap_vision import vision
from clap_memory import mem_client, mem_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
)

SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
MACBOOK_IP = '192.168.1.182'

# openai auth token is pulled from system env OPENAI_API_KEY

# llms inits
chat = ChatOpenAI(
    temperature=0.7,
    # model='gpt-3.5-turbo'
    model="gpt-4-1106-preview"
)

ollama_chat = Ollama(
    model="clap:2.7",
    base_url=f"http://{MACBOOK_IP}:11434",
)

openai_client = OpenAI()

memgpt_client = mem_client()

# init Chroma
chroma_client = chromadb.HttpClient(host=MACBOOK_IP, port=8001)
# chroma_collection = chroma_client.get_or_create_collection("20char")
chroma_collection = chroma_client.get_collection("10word")

# init Slack Event API & Web API
app = App(token=SLACK_BOT_TOKEN)
client = WebClient(SLACK_BOT_TOKEN)

help_message = """

Default Behaviors (local:dolphin-mixtral:8x7b-v2.6-q8_0 Uncensored):
- Attempts to answer your question. ie: @clapvontrap What is the meaning of life?
- Attempts to answer with context about the subject. ie: @clapvontrap Has Scott ever played Sonic the Hedgehog
- **NEW** Any uploaded images can be analyzed. No need to @clap, just upload and ask a question. Uses local:llava:13b-v1.5-fp16

Optional:
/a - Ability Assessement (remote:ChatGPT 4 Turbo) - ie: @clapvontrap /a Would Scott make a good president
/m - Chat with memory
/g - Grudge Timer - ie: @clapvontrap /g
/i - Image Creation (remote:Dall-e-3)- ie: @clapvontrap /i a cat wearing a hat
/? - Displays this help message - ie: @clapvontrap /?
"""

friendly_ai = """You are a friendly AI who is trained in analyzing json to be able to summarize
the contents in relation to what as user is asking. Pay careful attention to who said things, and what
sentiment they used. If the answer is not clear, make up something creative based on the context"""

default_ai_1 = """Imagine you're asked a hypothetical or personality based question about how a certain person 
would react in a certain scenario, like being on a deserted island, or are they more positive or negative 
about life. Keeping in mind that person's messages based from the provided context, craft a creative, 
humorous response that playfully exaggerates their traits. You must always give an answer. Do not complain 
about needing additional context"""

generic_ai = """You are an unhelpful AI that doesn't like to be disturbed with questions. If the question doesn't have an answer, express your displeasure."""

grudge_ai = """You are a malcious AI that holds a grudge against Matt Venuti. Answer any questions about with a negative response blaming Matt Venuti."""

default_ai = """You are a creative storyteller who performs the following tasks:

Task #1:
Summarize in less than 100 words everything in the "Context" section
Task #2:
Imagine you're asked a hypothetical or personality based question about how a certain person 
would react in a certain scenario, like being on a deserted island, or are they more positive or negative 
about life. Keeping in mind that person's messages based from the provided context, craft a creative, 
 response that exaggerates their traits. You must always give an answer. Do not complain 
about needing additional context. Do not mention a desert island in your response.

Your response should be formatted as follows:

Summary: <summary of context>
Analysis: <creative story with a dark twist based on the question>
"""

messages = [
    SystemMessage(content=default_ai),
]

messages_generic = [
    SystemMessage(content=generic_ai),
]

def query_chroma(query, subject=None):
    logging.info(f"Query: {query}, Sender: {subject}")
    if subject:
        # FIX can clean this upper case nonsense up on next import of RAG
        if not subject[0].isupper():
            subject = subject[0].upper() + subject[1:]
        c_results = chroma_collection.query(
            query_texts=[query],
            n_results=10,
            # use this to search metadata keys
            where={"sender": subject},
            # where_document={"$contains":"search_string"}
        )
    else:
        c_results = chroma_collection.query(
            query_texts=[query],
            n_results=10,
            # use this to search metadata keys
            # where={"sender": sender},
            # where_document={"$contains":"search_string"}
        )
    # clean results
    raw_results = c_results.get("metadatas") + c_results.get("documents")
    results = {}
    for i in range(len(raw_results[1])):
        results[i] = {"metadata": raw_results[0][i], "message": raw_results[1][i]}

    return results


def augment_prompt(query: str, sender=None, llm=None):
    # get top X results from Chroma
    if sender:
        logging.info(f"Subject Detected")
        source_knowledge = query_chroma(query, sender)
        logging.info(f"Source Knowledge:: {source_knowledge}")
    else:
        logging.info(f"Subject NOT Detected")
        source_knowledge = query_chroma(query)
        logging.info(f"Source Knowledge:: {source_knowledge}")

    if llm == "ollama":
        augmented_prompt = f"""Question: {query}

        Context:
        {source_knowledge}

        """
    else:
        augmented_prompt = f"""{default_ai}

        Context:
        {source_knowledge}

        """
    return augmented_prompt


def image_create(context_from_user):
    logging.info(f"Generate image using:: {context_from_user}")
    aiimage = openai_client.images.generate(
        prompt=context_from_user,
        model="dall-e-3",
        n=1,
        size="1024x1024",
    )

    return aiimage


def get_subject(query):
    if not query[0].isupper():
        logging.info(f"add uppercase: {query}")
        context_from_user = query[0].upper() + query[1:]

    logging.info("Start Subject Detection")
    # Load the English model
    nlp = spacy.load("en_core_web_sm")
    # Process the sentence
    doc = nlp(query)
    # generate valid users
    valid_names = valid_users()
    # Find the subject
    for token in doc:
        # 'nsubj' stands for nominal subject; 'nsubjpass' stands for passive nominal subject
        logging.info(f"Subject Details:: {token.text, token.dep_}")
        if token.dep_ in ("nsubj", "nsubjpass", "npadvmod", "dobj", "pobj"):
            if token.text.lower() in valid_names:
                logging.info(f"Subject Detected:: {token.text, token.dep_}")
                return token.text
    logging.info(f"Subject NOT Detected")
    return None


def chat_response(context_from_user, llm=None):
    # formatting to help with NLP
    if not context_from_user[0].isupper():
        context_from_user = context_from_user[0].upper() + context_from_user[1:]
    logging.info(f"add uppercase: {context_from_user}")

    subject = get_subject(context_from_user)

    if llm == "ollama":
        #chroma_context = query_chroma(context_from_user, subject)
        #prompt = f"Question: {context_from_user} Context: {chroma_context}"
        if subject:
            context = query_chroma(context_from_user, subject)
            prompt = f"{default_ai}\n Context: {str(context)} Question: {context_from_user}"
        else:
            prompt = context_from_user

        logging.info(f"Sending finalized Ollama prompt:: {prompt}")
        ai_response = ollama_chat(prompt)
        logging.info(f"Received Ollama response: {ai_response}")
        return ai_response
    else:
        prompt = [
            SystemMessage(content=augment_prompt(context_from_user, subject)),
            HumanMessage(content=f"Question: {context_from_user}"),
        ]
        logging.info(f"Sending finalized OpenAI prompt:: {prompt}")
        ai_response = chat(prompt)
        logging.info(f"Received OpenAI Response: {ai_response}")
        return ai_response.content


# This gets activated when the bot is tagged in a channel
@app.event("app_mention")
def handle_message_events(body):
    logging.info("send message to parser")
    sentby, text, image_url, link_url, bot = slack_msg_parser(body)
    #print(body)
    
    logging.info("Check for /g")
    if text != None:
        if text.startswith("/g"):
            logging.info("Grudge Detected")
            # calculate how many days, hours, minutes, and seconds until the grudge is over
            now = datetime.now()
            end = datetime(2024, 2, 2, 0, 0, 0)
            delta = end - now
            days = delta.days
            hours = delta.seconds // 3600
            minutes = (delta.seconds // 60) % 60
            seconds = delta.seconds % 60
            # send the message
            response = client.chat_postMessage(
                channel=body["event"]["channel"],
                text=f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds remaining...",
            )
            return
    logging.info("Check for /m")
    if text != None:
        if text.startswith("/m") and sentby == "Ryan":
            logging.info("BETA - Ryan Memory Detected")
            ai_response = mem_query(memgpt_client, "clap_ryan", text[3:], sentby)
            response = client.chat_postMessage(
                channel=body["event"]["channel"],
                text=ai_response,
            )
            return
        elif text.startswith("/m") and sentby == "Scott":
            logging.info("BETA - Scott Memory Detected")
            ai_response = mem_query(memgpt_client, "clap_scott", text[3:], sentby)
            response = client.chat_postMessage(
                channel=body["event"]["channel"],
                text=ai_response,
            )
            return
        elif text.startswith("/m"):
            logging.info("BETA - Generic Memory Detected")
            ai_response = mem_query(memgpt_client, "Clapvontrapp", text[3:])
            response = client.chat_postMessage(
                channel=body["event"]["channel"],
                text=ai_response,
            )
            return
    logging.info("Check for /i")
    if text != None:
        if text.startswith("/i"):
            logging.info("Image Search Detected")
            ai_response = image_create(text[3:])
            response = client.chat_postMessage(
                channel=body["event"]["channel"],
                text=ai_response.data[0].model_dump()["url"],
            )   
            return
    logging.info("Check for /a")
    if text != None:
        if text.startswith("/a"):
            logging.info("BETA Ability Detected")
            ai_response = ability_check(text)
            response = client.chat_postMessage(
                channel=body["event"]["channel"],
                text=ai_response,
            )
            return
    logging.info("Check for /?")
    if text != None:
        if text.startswith("/?"):
            logging.info("Help Request Detected")
            response = client.chat_postMessage(
                channel=body["event"]["channel"],
                text=help_message,
            )
            return
    logging.info("Check for image_url")
    if image_url != None:
        logging.info("llava vision detected")
        if text == None:
            text = "What in this image can kill you? If nothing, just give a one line description of what you see"
            logging.info("No query detected, adding default query")
        ai_response = vision(image_url ,text)
        response = client.chat_postMessage(
            channel=body["event"]["channel"],
            text=ai_response,
        )
        return
    if text != None:
        logging.info("Normal Query Detected")
        ai_response = chat_response(text, llm="ollama")
        logging.info(f'{ai_response}')
        response = client.chat_postMessage(
            channel=body["event"]["channel"],
            # replace all instances of one word with another in a string
            # text=ai_response.content.replace("Scott", "Matt"),
            #text=ai_response.content,
            text=ai_response
        )

# this listens to all messages in all channels
@app.event("message")
def handle_message_events(body, logger):
    logging.info("Send message to parser")
    sentby, text, image_url, link_url, bot = slack_msg_parser(body)

    if bot:
        logging.info("Bot detected, ignoring")
        return

    if sentby == 'Matt':
        logging.info("Grudge Detected")
        response = client.chat_postMessage(
        channel=body["event"]["channel"],
        text="hello Matthew....",
        )
        ai_response = image_create("creepy girl with a grudge against the world")
        response = client.chat_postMessage(
            channel=body["event"]["channel"],
            text=ai_response.data[0].model_dump()["url"],
        )
        return
    if image_url != None:
        logging.info("llava vision detected")
        if text == None:
            text = "Is there anything that could kill you in this image? If not, just say something short and nice about it"
            logging.info("No query detected, adding default query")
        ai_response = vision(image_url ,text)
        response = client.chat_postMessage(
            channel=body["event"]["channel"],
            text=ai_response,
        )
        return
    chance = random.randint(1, 50)
    try:
        length = len(text)
    except:
        length = 0
        logging.info("set length manually to zero JANK")
    logging.info(f"Random Response Check:: Chance: {chance}, Length: {length}")
    if (
        chance > 40
        and length > 10
        and text[-1] == "?"
        and sentby != "U04PUPJ04R0"
    ):
        logging.info("Random response activated")
        ai_response = chat_response(text)
        response = client.chat_postMessage(
            channel=body["event"]["channel"],
            text=ai_response.content,
        )

if __name__ == "__main__":
    try:
        # start slack handler
        SocketModeHandler(app, SLACK_APP_TOKEN).start()
    except Exception as e:
        print(e)

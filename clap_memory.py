from memgpt import MemGPT
from memgpt.config import AgentConfig
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
)

def mem_client():
    # Create a MemGPT client object (sets up the persistent state)
    client = MemGPT(
        auto_save=True,
        config={
            "model": "clap-mix:latest",
            "model_endpoint": "http://192.168.1.182:11434",
        #    "agent": agent_name,
        }
    )

    #if client.agent_exists("Clapvontrapp"):
    #    print("agent exists")
    
    logging.info(client.get_agent_config("Clapvontrapp"))

    return client



def webchat():
    url = "http://127.0.0.1:8283/agents/message"

    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    payload = {
        "user_id": "00000000000000000000cef89b01e14e",
        "stream": False,
        "user_id": "Scott",
        "agent_id": "Clapvontrapp",
        "message": "have i ever asked you about yourself?",
        "role": "user"
    }
    response = requests.post(url, json=payload, headers=headers)

    print(response.text)

    # You can set many more parameters, this is just a basic example

# create an agent for the person if they don't exist
def create_agent(mem_client, agent_name, human):
    agent_config = AgentConfig(
        name=f"{agent_name}",
        persona="p_clapvontrapp",
        human=f"{human}",
        model="clap-mix:latest",
        #model_endpoint_type="ollama",
        #model_endpoint="http://localhost:11434",
    )
    agent_state = mem_client.create_agent(agent_config=agent_config)


    # use an existing agent
    # agent_id = <agent_name>

    # Now that we have an agent_name identifier, we can send it a message!
    # The response will have data from the MemGPT agent
def mem_query(client, agent_name, text, sentby=None):
    logging.info(client.get_agent_config(agent_name))
    logging.info(f"Sending message to Memgpt")
    response = client.user_message(agent_id=agent_name, message=text)
    for r in response:
        logging.info(r)
        if r.get("assistant_message"):
            response = r['assistant_message']
    return response


if __name__ == "__main__":
    client = mem_client()
    create_agent(client, "clap_scott", "Scott")
    response = mem_query(client, "clap_ryan", "what color is my car?")
    print(response)
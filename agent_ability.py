import autogen
import os
import chromadb


def ability_check(question):
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    collection = chroma_client.get_or_create_collection(name="10word")

    config_list = [{"model": "gpt-4-1106-preview", "api_key": OPENAI_API_KEY}]

    def query_chroma(query, subject=None):
        c_results = collection.query(
            query_texts=[query],
            n_results=5,
            # use this to search metadata keys
            where={"sender": subject},
            # where_document={"$contains":"search_string"}
        )
        raw_results = c_results.get("metadatas") + c_results.get("documents")
        results = {}
        for i in range(len(raw_results[1])):
            results[i] = {"metadata": raw_results[0][i], "message": raw_results[1][i]}

        return results

    llm_config = {
        "config_list": config_list,
        "seed": 42,
        "temperature": 0.7,
        "functions": [
            {
                "name": "query_chroma",
                "description": "Looks up a query in the Chroma database and returns the results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "The subject of the hypothetical question",
                        },
                        "query": {
                            "type": "string",
                            "description": "A physical or mental abilities that are required to answer the hypothetical questions",
                        },
                    },
                    "required": ["query", "subject"],
                },
            },
        ],
    }

    # create a prompt for our agent
    claptrap_sys = f"""
    You are a helpful AI that will answer hypothetical questions and scenarios about people

    Task #1 - Generate a list of the top 3 best, creative mental or physical abilities, and top 3 worst, creative mental or physical abilities that would be required to answer: {question}
    Task #2 - For each ability, retrieve context from the Chroma database specific to that ability for the subject of the question. Also retrieve context for the specific topic of the question. Do not group them all in one query.
    Task #3 - Finally, using ALL the provided context, craft an imaginative
    response that calls out their abilities in order to answer: {question}. You must always give an answer. Do not complain 
    about needing additional context. Make sure to include quotes from the context in your response. The response should not be a list.

    Once you have completed task #3 output TERMINATE
    """

    # AGENT - create assistant
    claptrap = autogen.AssistantAgent(
        name="claptrap",
        system_message=claptrap_sys,
        llm_config=llm_config,
    )

    # AGENT - create user proxy
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=15,
        is_termination_msg=lambda x: x.get("content", "")
        and x.get("content", "").rstrip().endswith("TERMINATE"),
    )

    # function registration
    user_proxy.register_function(
        function_map={
            "query_chroma": query_chroma,
        }
    )

    def _reset_agents():
        claptrap.reset()
        user_proxy.reset()

    _reset_agents()

    # initiate conversation
    user_proxy.initiate_chat(claptrap, message=question)

    # print(user_proxy._oai_messages[claptrap][-1]['content'])
    for c in user_proxy._oai_messages[claptrap]:
        if c["content"]:
            if "TERMINATE" in c["content"]:
                print(c["content"])
                answer = c["content"].split("TERMINATE")[0]

    return answer


if __name__ == "__main__":
    try:
        answer = ability_check("Is Scott good at video games?")

    except Exception as e:
        print(e)

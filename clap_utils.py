import json
import logging
import spacy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
)

# conver
def clap_usermap(name=None, user=None):

    f = open("usermap.json")
    usermap = json.load(f)
    # lookup user, return name
    if user:
        name = usermap[user]
    # lookup name, return user
    if name:
        user = [k for k, v in usermap.items() if v == name][0]
    # return username wrapped in <@ and >
    if name:
        mention =  f"<@{user}>"

    return (name, user, mention)


def slack_msg_parser(body):
    #  finder
    # print(body)
    # ignore if the message is from a bot
    # clatrap is B04PE7UMAB1
    if "bot_id" in body["event"]:
        logging.info(f"Bot detected:: {body['event']['bot_id']}")
        bot = True
    else:
        bot = False

    try:
        if body["event"]["message_changed"]:
            logging.info("janky subtype detected")
            subtype = True
    except:
        subtype = False

    # do something if message is an image
    if "files" in body["event"]:
        logging.info(f"Image detected:: {body['event']['files'][0]['url_private_download']}")
        # get the url
        image_url = body["event"]["files"][0]["url_private_download"]
    else:
        image_url = None

    # do something if from_url is in body
    if subtype: # jank
        try:
            link_url = body["event"]["message"]["attachments"][0]["from_url"]
            logging.info(f"Link detected:: {link_url}")
        except:
            link_url = None
    else:
        link_url = None
    
    # get sentby
    if "user" in body["event"]:
        name, user, mention = clap_usermap(user=body["event"]["user"])
        sentby = name
        logging.info(f"Sentby:: {sentby}")
    else:
        sentby = None
    
    # do something if text in body
    if "text" in body["event"] and not subtype:
        # drop empty messages
        if len(body["event"]["text"]) ==  0:
            text = None
        else:
            text = body["event"]["text"]
            logging.info(f"Text detected:: {body['event']['text']}")
            # split text into a list
            text = text.split()
            # replace all mentions with the user's name, but leave the other text intact
            for i in text:
                if i.startswith("<@"):
                    name, user, mention = clap_usermap(user=i[2:-1])
                    if name == "clapvontrapp":
                        # prune index of clapvontrapp
                        text.pop(text.index(i))
                    else:
                        text[text.index(i)] = name

            # convert text back to a string
            text = " ".join(text)
            # capatalize the first letter of the message
            if len(text) > 0:
                if not text[0].isupper():
                    text = text[0].upper() + text[1:]
            else:
                text = None
            logging.info(f"Text cleaned:: {text}")
    else:
        text = None

    logging.info(f"Message parsed:: {sentby}, {text}, {image_url}, {link_url}, {bot}")
    return (sentby, text, image_url, link_url, bot)            

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

def chat_history():
    pass

def valid_users():
    file_path = "usermap.json"
    with open(file_path, "r") as file:
        data = json.load(file)
        values_list = list(data.values())
        values_list = [name.lower() for name in values_list]
    return values_list

if __name__ == "__main__":
    try:
        with open('slack_body.json') as f:
            test_slack = json.load(f)
        for i in test_slack:
            i = {"event": i}
            foo = slack_msg_parser(i)
            print(foo)

    except Exception as e:
        print(e)
        print('error')

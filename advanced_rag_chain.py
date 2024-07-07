from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import CharacterTextSplitter
from langchain.prompts import PromptTemplate
import os


# initialize LLM
llm = ChatOpenAI(model_name='gpt-4o', temperature=0.2, streaming=True)
embedding_model = OpenAIEmbeddings(disallowed_special=())

# summaries
summaries = []
for file in os.listdir('summaries'):
    if file[-4:] == '.txt':
        loader = TextLoader('summaries/' + file)
        summaries.extend(loader.load())
summary_splitter = CharacterTextSplitter(separator='<section>',
                                         chunk_size=500,
                                         chunk_overlap=200,
                                         length_function=len,
                                         is_separator_regex=False)
summaries = summary_splitter.split_documents(summaries)
summary_db = FAISS.from_documents(summaries, embedding=embedding_model)
summary_retriever = summary_db.as_retriever()

# documents
documents = []
for file in os.listdir('documents'):
    if file[-4:] == '.txt':
        loader = TextLoader('documents/' + file)
        documents.extend(loader.load())
document_splitter = CharacterTextSplitter(separator='<section>',
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len,
                                          is_separator_regex=False)
documents = document_splitter.split_documents(documents)
document_db = FAISS.from_documents(documents, embedding=embedding_model)
document_retriever = document_db.as_retriever()


# history
history_store = {}
def get_session_history(session_id: str):
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]
history = get_session_history('abc123')
history.add_ai_message('Hi! My name is JohnnAI. '
                       'John Rizcallah built me to help you get to know him a bit better. '
                       "You can ask me about him and I'll do my best to answer your questions! "
                       "How can I help you?")

# rephrasing query
with open('prompts/rephrasing_prompt.txt', 'r') as file:
    rephrasing_prompt = file.read()
rephrasing_template = PromptTemplate.from_template(rephrasing_prompt)


def get_rephrase_response(input: str):
    chat_history = get_session_history('abc123')
    history.add_user_message(input)
    prompt = rephrasing_template.format(input=input, chat_history=chat_history)
    response = llm.invoke(prompt).content
    return response


with open('prompts/rewrite_with_history.txt', 'r') as file:
    rewrite_with_history_prompt = file.read()
rewrite_with_history_template = PromptTemplate.from_template(rewrite_with_history_prompt)


def get_rewritten_prompt_with_history(input: str):
    chat_history = get_session_history('abc123')
    prompt = rewrite_with_history_template.format(input=input, chat_history=chat_history)
    response = llm.invoke(prompt).content
    return response


with open('prompts/router_prompt.txt', 'r') as file:
    router_prompt = file.read()
router_template = PromptTemplate.from_template(router_prompt)


def get_router_decision(input: str):
    prompt = router_template.format(input=input)
    response = llm.invoke(prompt).content
    return response


with open('prompts/depth_prompt.txt', 'r') as file:
    depth_prompt = file.read()
depth_template = PromptTemplate.from_template(depth_prompt)


def get_depth_decision(input: str):
    prompt = depth_template.format(input=input)
    response = llm.invoke(prompt).content
    return response


with open('prompts/question_prompt.txt', 'r') as file:
    question_prompt = file.read()
question_template = PromptTemplate.from_template(question_prompt)


def get_rewritten_question(input, context):
    prompt = question_template.format(input=input, context=context)
    response = llm.invoke(prompt).content
    return response


with open('prompts/final_prompt.txt', 'r') as file:
    final_prompt = file.read()
final_template = PromptTemplate.from_template(final_prompt)

def get_final_response(input: str, context: str):
    prompt = final_template.format(input=input, context=context)
    response = llm.invoke(prompt).content
    return response


def parse_yaml_response(yaml: str):
    if 'true' in yaml:
        return True
    elif 'false' in yaml:
        return False
    else:
        return None

def full_chain(input):
    # decide whether to rephrase the question to include chat history
    rephrase_response = parse_yaml_response(get_rephrase_response(input))
    if rephrase_response:
        rewritten_input = get_rewritten_prompt_with_history(input)
    else:
        rewritten_input = input

    # decide whether to rewrite the question using RAG context
    router_response = parse_yaml_response(get_router_decision(rewritten_input))
    if router_response:
        # decide whether to rewrite the question using summaries or documents
        depth_response = parse_yaml_response(get_depth_decision(rewritten_input))
        if depth_response:
            context = document_retriever.invoke(rewritten_input)
        else:
            context = summary_retriever.invoke(rewritten_input)
        context = [i.page_content for i in context]
        context = '...'.join(context)
        rewritten_input = get_rewritten_question(rewritten_input, context)

    else:
        context = ''

    final_response = get_final_response(rewritten_input, context)
    return final_response


if __name__ == '__main__':
    print(full_chain('What do you think of John Rizcallah?'))
    print('\n')
    print(full_chain('Tell me about his sisters.'))
    print('\n')
    print(full_chain('What was the last question I asked?'))
    print('\n')
    print(full_chain('Can you give me the gist of your response again?'))
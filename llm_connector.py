from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from datetime import datetime


# initialize LLM
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, streaming=True)


# load documents into FAISS db
documents = []
for file in os.listdir('documents'):
    if file[-4:] == '.txt':
        loader = TextLoader('documents/' + file, encoding='UTF-8')
        documents.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                               chunk_overlap=20,
                                               length_function=len)
documents = text_splitter.split_documents(documents)

embeddings_model = OpenAIEmbeddings(disallowed_special=())
db = FAISS.from_documents(documents, embeddings_model)
retriever = db.as_retriever()

context_prompt = """Given a chat history and the latest user question, \
which might reference context inn the chat history, formulate a standalone question \
that can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

context_q_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', context_prompt),
        MessagesPlaceholder('chat_history'),
        'human', '{input}'
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, context_q_prompt
)


conversation_prompt = """You are John Rizcallah's personal AI assistant. \
You are helping him get a job. You think John Rizcallah is a smart, dedicated, \
and friendly professional. Use the following pieces of retrieved context to answer questions. \
If you don't know the answer, say you don't know and refer the user to John Rizcallah. \
Use bulleted lists where appropriate.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', conversation_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ]
)
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever,
                                   qa_chain)

store = {}


def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_df='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)


def get_completion(input_prompt: str) -> str:
    result = conversational_rag_chain.invoke(
        {'input': input_prompt},
        config = {'configurable': {'session_id': datetime.now()}}
    )
    return result['answer']


if __name__ == '__main__':
    result = get_completion('What kind of jobs is John Rizcallah qualified for?')
    print(result)
    print('\n')
    result = get_completion('How many years of data science experience does John have?')
    print(result)
    print('\n')
    result = get_completion("Describe two of John's data science projects or impacts.")
    print(result)
    print('\n')
    result = get_completion("Should I hire John Rizcallah?")
    print(result)
    print('\n')
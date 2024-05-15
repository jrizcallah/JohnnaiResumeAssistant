import streamlit as st
from llm_connector import get_completion


st.set_page_config(page_title="JohnnAI - John Rizcallah's AI Resume Assistant")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {
            'role': 'assistant',
            'content': 'Hi! My name is JohnnAI. '
                       'John Rizcallah built me to help you get to know him a bit better. '
                       "You can ask me about him and I'll do my best to answer your questions!"
                       "How can I help you?"
        }
    ]

# display chat messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

# function to generate responses
def generate_response(input_prompt):
    response = get_completion(input_prompt)
    return response


# user-provided prompt
if prompt := st.chat_input(disabled=False):
    st.session_state.messages.append(
        {
            'role': 'user',
            'content': prompt
        }
    )
    with st.chat_message('user'):
        st.write(prompt)

if st.session_state.messages[-1]['role'] != 'assistant':
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            response = generate_response(prompt)
            st.write(response)

    message = {'role': 'assistant',
               'content': response}
    st.session_state.messages.append(message)

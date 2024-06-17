from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import re
import os


with open('prompts/summarization_prompt.txt') as file:
    summarization_prompt_template = file.read()


def split_sections(text: str, section_start='<section>', section_end='</section>') -> list[str]:
    sections = re.findall(text, section_start + '(.*)' + section_end)
    if not sections:
        return [text]
    return sections


def split_on_length(text: str, max_length=1000) -> list[str]:
    split_text = [text[i: i + max_length] for i in range(0, len(text), max_length)]
    return split_text


def split_document(doc: str) -> list[str]:
    sections = split_sections(doc)
    split_text = []
    for sect in sections:
        split_text.extend(split_on_length(sect))
    return split_text


def summarize_document(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        doc = file.read()
    split_text = split_document(doc)
    summarization = ''
    for text in split_text:
        response = llm_chain.invoke(text)
        summarization += ' ' + response.content
    return summarization


llm = ChatOpenAI(model='gpt-4o')
prompt = PromptTemplate(template=summarization_prompt_template, input_variables=['text'])
llm_chain = prompt | llm


def summarize_all_files(dir_path: str):
    for file in os.listdir(dir_path):
        if file[-4:] == '.txt':
            print(f'\tSummarizing {file}...')
            summary = summarize_document(dir_path + file)
            with open('summaries/' + file.replace('.txt', '_summary.txt'),
                      'w', encoding='utf-8') as new_file:
                new_file.write(summary)

if __name__ == '__main__':
    test_output = summarize_document('utils/test_text.txt')
    print('Test Output: \n')
    print(test_output)

    print('Summarizing files now...')
    summarize_all_files('documents/')
    print('Done!')


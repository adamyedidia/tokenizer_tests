from openai import OpenAI

import tiktoken
import functools
import time
import string
from functools import cache

from functools import cache
from langchain_openai import ChatOpenAI


@cache
def get_client(model: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


def get_openai_response(prompt: str, model: str = 'gpt-4', temperature: float = 0.0) -> str:
    return  get_client(model=model, temperature=temperature).invoke(prompt).content

def retryable(max_retries=3, delay=0.1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        print(f"Attempt {attempt + 1} failed with error: {e}. No more retries.")
                        raise e
        return wrapper
    return decorator


@retryable()
def repetition_test(token: str) -> bool:
    DUMMY_TOKEN = 'Hello'
    string_to_repeat = f'{DUMMY_TOKEN}{token}'

    response_content = get_openai_response(
        f"Please repeat the following string back to me exactly and in its entirety. "
        f"The string need not make sense. Here is the string: {string_to_repeat}")

    if string_to_repeat in response_content or f'{DUMMY_TOKEN} {token}' in response_content:
        return True
    print(f'Failed repetition test: needed to repeat "{string_to_repeat}", but gave the response "{response_content}" instead.')
    return False


@retryable()
def spelling_test(token: str) -> bool:
    spelled_string = '-'.join([c for c in token])

    response_content = get_openai_response(
        f"Please spell out the following string exactly, with letters separated by '-' characters: {spelled_string}")

    if spelled_string in response_content or '-'.join([c for c in token if c != ' ']) in response_content:
        return True
    print(f'Failed repetition test: needed to spell out "{token}", but gave the response "{response_content}" instead.')
    return False    


def main():
    num_tokens = 0
    token_index = 98_000
    encoding = tiktoken.get_encoding("cl100k_base")

    repetition_failures = []
    spelling_failures = []

    while True:
        token = encoding.decode([token_index])
        do_not_test = False
        for c in token:
            if c not in f' {string.ascii_letters}':
                do_not_test = True

        if do_not_test:
            token_index += 1
            continue

        print(f'Testing {token} at index {token_index}...')
        if not repetition_test(token):
            repetition_failures.append(token)
        if not spelling_test(token):
            spelling_failures.append(token)

        token_index += 1
        if token_index % 100 == 0:
            print(f'Repetition failures: {repetition_failures}')
            print(f'Spelling failures: {spelling_failures}')

        if token_index > 100_000:
            break

if __name__ == '__main__':
    main()

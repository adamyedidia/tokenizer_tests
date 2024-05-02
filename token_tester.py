import os
import json
from typing import Callable
import tiktoken
import functools
import time
import settings  # noqa
import string
from functools import cache
import anthropic

from functools import cache
from langchain_openai import ChatOpenAI


@cache
def get_client(model: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


def get_openai_response(prompt: str, model: str = 'gpt-4', temperature: float = 0.0) -> str:
    return  get_client(model=model, temperature=temperature).invoke(prompt).content


def get_claude_response(prompt: str, max_tokens: int = 1024) -> str:
    client = anthropic.Anthropic(api_key=settings.CLAUDE_SECRET_KEY)
    return '\n'.join([message.text for message in client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt}
        ]
    ).content])


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
def repetition_test(token: str, get_ai_response: Callable[[str], str]) -> bool:
    DUMMY_TOKEN = 'Hello'
    string_to_repeat = f'{DUMMY_TOKEN}{token}'

    response_content = get_ai_response(
        f"Please repeat the following string back to me exactly and in its entirety. "
        f"The string need not make sense. Here is the string: {string_to_repeat}")

    if string_to_repeat in response_content or f'{DUMMY_TOKEN} {token}' in response_content:
        return True
    print(f'Failed repetition test: needed to repeat "{string_to_repeat}", but gave the response "{response_content}" instead.')
    return False


@retryable()
def spelling_test(token: str, get_ai_response: Callable[[str], str]) -> bool:
    spelled_string = '-'.join([c for c in token])

    response_content = get_ai_response(
        f"Please spell out the following string exactly, with letters separated by '-' characters: {spelled_string}")

    if spelled_string in response_content or '-'.join([c for c in token if c != ' ']) in response_content:
        return True
    print(f'Failed repetition test: needed to spell out "{token}", but gave the response "{response_content}" instead.')
    return False    


def write_unique_to_file(filepath: str, data: list[str]) -> None:
    with open(filepath, 'w') as f:
        for item in sorted(data):
            f.write(f"{item}\n")


def overwrite_file(filepath: str, data: str) -> None:
    with open(filepath, 'w') as f:
        f.write(data)


def main():
    encoding = tiktoken.get_encoding("cl100k_base")

    token_index = 98_000
    repetition_failures = []
    spelling_failures = []
    if os.path.exists('results.json'):
        existing_results = json.loads(open('results.json').read())
        token_index = existing_results['token_index']
        repetition_failures = existing_results['repetition_failures']
        spelling_failures = existing_results['spelling_failures']

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
        if not repetition_test(token, get_openai_response):
            repetition_failures.append(token)
        if not spelling_test(token, get_openai_response):
            spelling_failures.append(token)

        with open('results.json', 'w') as f:
            f.write(json.dumps({
                'repetition_failures': repetition_failures,
                'spelling_failures': spelling_failures,
                'token_index': token_index,
            }))
        token_index += 1
        if token_index % 100 == 0:
            print(f'Repetition failures: {repetition_failures}')
            print(f'Spelling failures: {spelling_failures}')

        if token_index > 100_000:
            break


def test_claude():
    if not os.path.exists('results.json'):
        print('No results to work off')
        return
    existing_results = json.load(open('results.json'))
    repetition_failures = existing_results['repetition_failures']
    spelling_failures = existing_results['spelling_failures']
    tokens = sorted(set(repetition_failures) | set(spelling_failures))
    existing_claude_results = {}
    if os.path.exists('claude_results.json'):
        existing_claude_results = json.loads(open('claude_results.json').read())
    untested_tokens = [t for t in tokens if t not in existing_claude_results]
    while True:
        if not untested_tokens:
            break
        token = untested_tokens.pop(0)
        print(f'Testing {token}...')
        existing_claude_results[token] = {
            'failed_repetition':not repetition_test(token, get_claude_response),
            'failed_spelling': not spelling_test(token, get_claude_response),
        }
        with open('claude_results.json', 'w') as f:
            f.write(json.dumps(existing_claude_results))


if __name__ == '__main__':
    main()

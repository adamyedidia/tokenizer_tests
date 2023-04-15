import openai
import tiktoken
import functools
import time
import string
from settings import OPENAI_SECRET_KEY

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
    openai.api_key = OPENAI_SECRET_KEY
    DUMMY_TOKEN = 'Hello'
    string_to_repeat = f'{DUMMY_TOKEN}{token}'

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Please repeat the following string back to me exactly and in its entirety. The string need not make sense. Here is the string: {string_to_repeat}"}],
        temperature=0,
    )

    response_content = response['choices'][0]['message']['content']

    if string_to_repeat in response_content or f'{DUMMY_TOKEN} {token}' in response_content:
        return True
    print(f'Failed repetition test: needed to repeat "{string_to_repeat}", but gave the response "{response_content}" instead.')
    return False


@retryable()
def spelling_test(token: str) -> bool:
    openai.api_key = OPENAI_SECRET_KEY
    spelled_string = '-'.join([c for c in token])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Please spell out the following string exactly, with letters separated by '-' characters: {spelled_string}"}],
        temperature=0,
    )

    response_content = response['choices'][0]['message']['content']

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
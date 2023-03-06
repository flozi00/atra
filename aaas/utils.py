import time
from functools import wraps, cache
import requests


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


@cache
def get_mail_from_google(auth_token):
    headers = {
        "authority": "content-people.googleapis.com",
        "accept": "*/*",
        "accept-language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "authorization": auth_token,
    }

    params = {
        "sources": "READ_SOURCE_TYPE_PROFILE",
        "personFields": "emailAddresses",
    }

    response = requests.get(
        "https://content-people.googleapis.com/v1/people/me",
        params=params,
        headers=headers,
    ).json()["emailAddresses"][0]["value"]

    return response

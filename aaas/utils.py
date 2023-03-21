import time
from functools import wraps, cache
import requests
from datetime import datetime
import logging


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        logging.debug(
            f"Function {func.__name__} Took {total_time:.4f} seconds at {dt_string}"
        )
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

    try:
        response = requests.get(
            "https://content-people.googleapis.com/v1/people/me",
            params=params,
            headers=headers,
        ).json()["emailAddresses"][0]["value"]

        return response
    except:
        return None


def check_valid_auth(request):
    token = request.request.headers.get("Authorization", "")
    if len(token) > 10:
        mail = get_mail_from_google(token)
        if mail is not None:
            return mail

    return False

import urllib.request
import os

BACKENDS = []

inference_only = os.getenv("inference_only", "False")
inference_only = True if inference_only == "True" else False

connect_to_master = os.getenv("connect_to_master", "True")
connect_to_master = True if connect_to_master == "True" else False

master_node = os.getenv("master_node", "")
master_user = os.getenv("master_user", "")
master_pass = os.getenv("master_pass", "")

fallback_url = os.getenv("fallback_url", "http://127.0.0.1:7860")

port_to_listen = os.getenv("PORT", 7860)

def health_check(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status/") as response:
            content = response.read().decode("utf-8")
            if "success" in content:
                return True
            else:
                return False
    except Exception as e:
        print(port, e)
        return False


async def increase_queue(port):
    global BACKENDS
    for x in range(len(BACKENDS)):
        if BACKENDS[x]["port"] == port:
            BACKENDS[x]["requests"] += 1


async def decrease_queue(port):
    global BACKENDS
    for x in range(len(BACKENDS)):
        if BACKENDS[x]["port"] == port:
            BACKENDS[x]["requests"] -= 1


def get_best_node(premium=False):
    global BACKENDS
    BACKENDS = sorted(BACKENDS, key=lambda d: d["requests"])
    print(BACKENDS)
    for x in range(len(BACKENDS)):
        if health_check(BACKENDS[x]["port"]) == True:
            return BACKENDS[x]["port"]
        else:
            BACKENDS.remove(BACKENDS[x])
            return get_best_node(premium)

    return False


def get_used_ports():
    global BACKENDS
    return [port["port"] for port in BACKENDS]

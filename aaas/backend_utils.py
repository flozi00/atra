import urllib.request
import os

BACKENDS = []

inference_only = os.getenv("inference_only", "False")
inference_only = True if inference_only == "True" else False

master_node = os.getenv("master_node", "")
master_user = os.getenv("master_user", "")
master_pass = os.getenv("master_pass", "")


def health_check(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status/") as response:
            content = response.read().decode("utf-8")
            if "success" in content:
                return True
            else:
                return False
    except:
        return False

async def increase_queue(port):
    global BACKENDS
    for x in range(len(BACKENDS)):
        if(BACKENDS[x]["port"] == port):
            BACKENDS[x]["requests"] += 1

async def decrease_queue(port):
    global BACKENDS
    for x in range(len(BACKENDS)):
        if(BACKENDS[x]["port"] == port):
            BACKENDS[x]["requests"] -= 1

def get_best_node(premium = False):
    global BACKENDS
    BACKENDS = sorted(BACKENDS, key=lambda d: d['requests']) 
    for x in range(len(BACKENDS)):
        if(premium == True):
            if(BACKENDS[x]["device"] == "gpu"):
                if(health_check(BACKENDS[x]["port"]) == True):
                    return BACKENDS[x]["port"]
                else:
                    BACKENDS.remove(BACKENDS[x])
                    return get_best_node(premium)

        if(BACKENDS[x]["device"] == "cpu"):
            if(health_check(BACKENDS[x]["port"]) == True):
                return BACKENDS[x]["port"]
            else:
                BACKENDS.remove(BACKENDS[x])
                return get_best_node(premium)

    return 7860

def get_used_ports():
    global BACKENDS
    return [port["port"] for port in BACKENDS]
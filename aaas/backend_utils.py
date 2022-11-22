import urllib.request
import os

CPU_BACKENDS = []
GPU_BACKENDS = []

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


def check_nodes():
    global CPU_BACKENDS, GPU_BACKENDS
    to_remove = []
    ports = GPU_BACKENDS + CPU_BACKENDS
    for port in ports:
        if health_check(port) == True:
            continue
        else:
            to_remove.append(port)

    for p in to_remove:
        if p in GPU_BACKENDS:
            GPU_BACKENDS.remove(p)
        if p in CPU_BACKENDS:
            CPU_BACKENDS.remove(p)

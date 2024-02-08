#!/bin/bash
args_array=("$@")
eval "granian --interface asgi $@:app"
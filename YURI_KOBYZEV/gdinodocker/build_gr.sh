#!/bin/bash

sudo docker build --network=host -t smokey -f Dockerfile "$@" .

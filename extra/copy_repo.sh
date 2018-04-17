#!/bin/bash
# Copy local git repository into docker - run with pull

name=$1

docker cp ../../WeGAN $name:/home/rafajdus

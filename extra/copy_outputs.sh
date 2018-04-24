#!/bin/bash
# Copy relevant stuff from docker to local -- run after experiments with scp to your env

name=$1
exp=$2

docker cp $name:/home/rafajdus/experiments/$exp/logs ../../experiments/$exp 
docker cp $name:/home/rafajdus/experiments/$exp/samples ../../experiments/$exp
docker cp $name:/home/rafajdus/WeGAN/experiments/$exp ../experiments/$exp

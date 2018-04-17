#!/bin/bash
# Copy dataset into docker - run only once

name=$1

docker cp ../../data_parsed/64x64 $name:/home/rafajdus/data_parsed/

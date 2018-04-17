#!/bin/bash
# Copy dataset into docker - run only once

name=$1

docker cp ../../data_parsed/32x32 $name:/home/rafajdus/data_parsed/

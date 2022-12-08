#!/bin/bash

mkdir models tuner runs

cd data
unzip target-data.zip
cd ..

# pip install keras_tuner
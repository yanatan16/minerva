#!/usr/bin/env bash

pushd /home/jon/Code/Minerva/pysandbox/experiments/data
./getStockData.py -o nasdaq_full_1990_to_2010_2 -s csv -b 19900101 -e 20101231 -i symbols/nasdaq.csv 
popd

#!/bin/bash
cd src/lib/models/networks/DCNv2_new sh
sh make.sh
cd ../../../..
python script.py -mp ../models/all_dla34.pth -vp ../videos/MOT16-03.mp4 -od ../results


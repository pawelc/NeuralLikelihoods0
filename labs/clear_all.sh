#!/bin/bash
find . -name best*pkl -exec rm {} \;
find . -name tensorboard -exec rm -fr {} \;
find . -name tensorboard_best -exec rm -fr {} \;
find -name "*.log" -exec rm {} \;


#!/bin/bash

srun -p interactive --time=04:00:00 --mem=16G --gres=gpu:v100:1 --pty bash
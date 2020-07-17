# -*- coding: utf-8 -*-

"""
Given a bunch of commands to run, check the available GPUs and run them on the GPUs in separate tmux sessions.
Usage: Just modify the settings in the Config class and then run python3 gpu_run.py
"""

import GPUtil
import subprocess
import sys
import time

slurm_target = sys.argv[1]
if not slurm_target:
    print("Usage: python3 fname.py slurm_target")
    exit()

class Config:
    """
    Global class that houses all configurations
    """
    
    # Shared args to put onto all of the JOBS
    SHARED_ARGS = ""

    # Specifies tasks to run. It maps tmux session name to the command to run in that session.
    JOBS = {
        # "distort_0" : "python3 distort_imagenet_instance_ResNet_3.py --total-workers=4 --worker-number=0",
        # "distort_1" : "python3 distort_imagenet_instance_ResNet_3.py --total-workers=4 --worker-number=1",
        # "distort_2" : "python3 distort_imagenet_instance_ResNet_3.py --total-workers=4 --worker-number=2",
        # "distort_3" : "python3 distort_imagenet_instance_ResNet_3.py --total-workers=4 --worker-number=3",

        "imagenet_vgg16_tune_styleLossLambda_2e-2_ImageNetR_classes_lr1e-3_epochs30": "python3 tune_imagenet_distorted.py \
            --data-standard=/data/imagenet/train/ \
            --data-val=/data/imagenet/val/ \
            --save=checkpoints/imagenet_vgg16_tune_styleLossLambda_2e-2_ImageNetR_classes_lr1e-3_epochs30 \
            --savedir-model=/data2/sauravkadavath/imagenet-classifier-checkpoints/imagenet_vgg16_tune_styleLossLambda_2e-2_ImageNetR_classes_lr1e-3_epochs30 \
            --arch=vgg16 \
            --lr=0.001 \
            --style-loss-lambda=0.02 \
            --pretrained \
            --batch-size=128 \
            --epochs=30",

        "imagenet_vgg16_tune_styleLossLambda_0_ImageNetR_classes_lr1e-3_epochs30": "python3 tune_imagenet_distorted.py \
            --data-standard=/data/imagenet/train/ \
            --data-val=/data/imagenet/val/ \
            --save=checkpoints/imagenet_vgg16_tune_styleLossLambda_0_ImageNetR_classes_lr1e-3_epochs30 \
            --savedir-model=/data2/sauravkadavath/imagenet-classifier-checkpoints/imagenet_vgg16_tune_styleLossLambda_0_ImageNetR_classes_lr1e-3_epochs30 \
            --arch=vgg16 \
            --lr=0.001 \
            --style-loss-lambda=0.0 \
            --pretrained \
            --batch-size=128 \
            --epochs=30"
    }

    # Time to wait between putting jobs on GPUs (in seconds). This is useful because it might take time 
    # for a process to actually load the network onto the GPU, so we wait until that is done before 
    # selecting the GPU for the next process.
    SLEEP_TIME = 1

    SLURM_HEADER = "srun --pty -p gpu_jsteinhardt -w {0} -c 10 --gres=gpu:1".format(slurm_target)

# Stick the shared args onto each JOB
for key, value in Config.JOBS.items():
    new_value = value + " " + Config.SHARED_ARGS
    Config.JOBS[key] = new_value


for index, (tmux_session_name, command) in enumerate(Config.JOBS.items()):
    # Get the best available GPU
    print("Running \"{0}\" with SLURM".format(command))

    result = subprocess.run("tmux new-session -d -s {0}".format(tmux_session_name), shell=True)        
    if result.returncode != 0:
        print("Failed to create new tmux session called {0}".format(tmux_session_name))
        sys.exit(result.returncode)

    result = subprocess.run("tmux send-keys '{0} {1}' C-m".format(
        Config.SLURM_HEADER, command
    ), shell=True)
    if result.returncode != 0:
        print("Failed to run {0} in tmux session".format(command, tmux_session_name))
        sys.exit(result.returncode)

    print("---------------------------------------------------------------")

    if index < len(Config.JOBS) - 1:
        time.sleep(Config.SLEEP_TIME)

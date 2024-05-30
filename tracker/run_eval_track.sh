#!/bin/bash

base_dir="/home/wardlewo/Reggie/data/20240222_urchin_videos_aims/video/unseen_test"
run_num=3

find "$base_dir" -type f -name "*.MP4" | while read -r f
do
    echo "$f"
    python eval_tracks.py "$f" "$run_num"
done
    

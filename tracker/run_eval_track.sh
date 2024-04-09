#!/bin/bash

base_dir="/home/java/Java/data/20231201_urchin/sorted_videos/unseen_test"

find "$base_dir" -type f -name "*.MP4" | while read -r f
do
    echo "$f"
    python eval_tracks.py "$f"
done
    

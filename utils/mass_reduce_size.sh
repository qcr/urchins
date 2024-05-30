#!/bin/bash

base_dir="/home/wardlewo/Reggie/data/20240402_combined_model_data/images/test"
out_dir="/home/wardlewo/Reggie/data/urchin_data_output/test_results"

find "$base_dir" -type f -name "*.jpg" | while read -r f
do
    echo "$f"
    python demo_tools/reduce_image_size.py "$f" "$out_dir"
done
    

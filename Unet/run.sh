#!/usr/bin/env bash
task_id="$1"
file_path="$2"
ext_cont="$3"
echo "$task_id"
echo "$file_path"
echo "$ext_cont"
source activate inpainting_detection
python /home/dell/soft/ljc_methods/video_inpainting_detection/unet/main.py "$task_id" "$file_path" "$ext_cont"
conda deactivate
# bash /home/dell/soft/ljc_methods/video_inpainting_detection/unet/run.sh 001 /home/dell/soft/ljc_methods/video_inpainting_detection/unet/samples/ '{"JSON_FILE_PATH": "/home/dell/soft/ljc_methods/video_inpainting_detection/unet/call_unet.json", "TMP_DIR": "/home/dell/soft/ljc_methods/video_inpainting_detection/unet/temp/", "GPU_ID": 0}'
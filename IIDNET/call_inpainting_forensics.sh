#!/usr/bin/env bash
task_id="$1"
file_path="$2"
ext_cont="$3"
echo "$task_id"
echo "$file_path"
echo "$ext_cont"
source activate open-mmlab
python /home/dell/soft/ljc_methods/video_inpainting_detection/iid_net/main.py "$task_id" "$file_path" "$ext_cont"
conda deactivate
# bash /home/dell/soft/ljc_methods/video_inpainting_detection/iid_net/call_inpainting_forensics.sh 001 /home/dell/soft/ljc_methods/video_inpainting_detection/iid_net/test/ '{"JSON_FILE_PATH": "/home/dell/soft/ljc_methods/video_inpainting_detection/iid_net/call_inpainting_forensics.json", "TMP_DIR": "/home/dell/soft/ljc_methods/video_inpainting_detection/iid_net/temp/", "GPU_ID": 0}'
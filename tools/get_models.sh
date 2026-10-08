#!/bin/bash
# Downloads the YuNet face detector next to the scripts. The ASD weights come from the Light-ASD repo (see setup_mac_native.sh).
HERE="$(cd "$(dirname "$0")" && pwd)"
curl -L -o "$HERE/yunet.onnx" https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
echo "Put a bold font at $HERE/Mv.ttf (Montserrat Bold works well)."

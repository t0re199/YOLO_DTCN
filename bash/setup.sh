#!/bin/bash

YOLO_ROOT=".yolov3"
YOLO_WEIGHTS_DIR="$YOLO_ROOT/weights"
YOLO_CFG_DIR="$YOLO_ROOT/cfg"

cd ..

mkdir $YOLO_DIR
mkdir $YOLO_WEIGHTS_DIR
mkdir .yolov3/cfg

echo "[*] Downloading Yolo Weights"
wget "https://pjreddie.com/media/files/yolov3-tiny.weights" -P $YOLO_WEIGHTS_DIR &> /dev/null
if [[ $? - ne 0 ]]; then
{
    echo "Unable to Download Yolo weights" > &2
    exit 0x1
}
fi
echo "[*] Downloading Yolo Configs"

wget "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg" -P $YOLO_CFG_DIR &> /dev/null
if [[ $? - ne 0 ]]; then
{
    echo "Unable to Download Yolo Config file" > &2
    exit 0x2
}
fi

echo "[*] Done."

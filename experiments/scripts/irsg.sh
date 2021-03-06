#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  irsg_objs)
    TRAIN_IMDB="irsg_train_objs"
    TEST_IMDB="irsg_val_objs"
    PT_DIR="irsg_objs"
    ITERS=70000
    ;;
  irsg_attrs)
    TRAIN_IMDB="irsg_train_attrs"
    TEST_IMDB="irsg_val_attrs"
    PT_DIR="irsg_attrs"
    ITERS=70000
    ;;
  irsg_objs_smol)
    TRAIN_IMDB="irsg_train_objs_smol"
    TEST_IMDB="irsg_val_objs_smol"
    PT_DIR="irsg_objs_smol"
    ITERS=70000
    ;;
  irsg_attrs_smol)
    TRAIN_IMDB="irsg_train_attrs_smol"
    TEST_IMDB="irsg_val_attrs_smol"
    PT_DIR="irsg_attrs_smol"
    ITERS=70000
    ;;
  glasses)
    TRAIN_IMDB="glasses_train"
    TEST_IMDB="glasses_test"
    PT_DIR="glasses"
    ITERS=70000
    ;;
  umbrella)
    TRAIN_IMDB="umbrella_train"
    TEST_IMDB="umbrella_test"
    PT_DIR="umbrella"
    ITERS=70000
    ;;
  kite)
    TRAIN_IMDB="kite_train"
    TEST_IMDB="kite_test"
    PT_DIR="kite"
    ITERS=70000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver.prototxt \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TRAIN_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

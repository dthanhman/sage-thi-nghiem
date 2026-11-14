#!/bin/sh
# Train SegFormer-B0 cho sage rồi vis (100 epoch, seed 0), chấm test sau mỗi model.
# Chạy lại bao nhiêu lần cũng được:
#   - model đã train xong (có last.pt) thì bỏ qua train;
#   - model đang dở (có resume.pt) thì train tiếp từ cuối epoch gần nhất;
#   - đã chấm test (có test_ket_qua.json) thì bỏ qua chấm.
# workers=2 để đỡ tốn RAM (máy 11,7 GB). Không chạy việc nặng khác song song.
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=/d/Miniconda3/envs/DoAn/python.exe
rm -rf runs/sage_b0_s99   # bản chạy thử 1 epoch
for bt in sage vis; do
  r=runs/${bt}_b0_s0
  if [ ! -f $r/last.pt ]; then
    echo "train $bt (log: runs_${bt}_train.log)"
    $PY train_seg.py --bien_the $bt --model b0 --epochs 100 --seed 0 --workers 2 --tiep >> runs_${bt}_train.log 2>&1
  fi
  if [ ! -f $r/test_ket_qua.json ]; then
    $PY eval_seg.py --run $r --luu_anh > runs_${bt}_eval.log 2>&1
  fi
  echo "xong $bt"
done

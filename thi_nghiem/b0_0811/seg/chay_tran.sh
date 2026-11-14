#!/bin/sh
# Thí nghiệm trần B0: tỉ lệ IR cố định 30% và 70% (50% = tb, đã có 2 seed). Seed 0 trước.
# Chạy lại bao nhiêu lần cũng được: model xong thì bỏ qua, model dở thì train tiếp (resume.pt).
# workers=2 để đỡ tốn RAM. Không chạy việc nặng khác song song. Đừng bấm Ctrl+C.
# Thêm seed: sh chay_tran.sh ir30:1 ir70:1
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=/d/Miniconda3/envs/DoAn/python.exe
for bt in ir30 ir70; do
  if [ "$(ls anh/$bt/test 2>/dev/null | wc -l)" -lt 280 ] || [ "$(ls anh/$bt/train 2>/dev/null | wc -l)" -lt 1220 ]; then
    echo "xuất ảnh $bt"
    $PY xuat_anh.py --bien_the $bt > xuat_$bt.log 2>&1
  fi
done
DS_VIEC="${*:-ir30:0 ir70:0}"
for viec in $DS_VIEC; do
  bt=${viec%:*}; s=${viec#*:}
  r=runs/${bt}_b0_s${s}
  if [ ! -f $r/last.pt ]; then
    echo "train $bt seed $s (log: runs_${bt}_s${s}_train.log) – khoảng 1,5–3 giờ"
    $PY train_seg.py --bien_the $bt --model b0 --epochs 100 --seed $s --workers 2 --tiep >> runs_${bt}_s${s}_train.log 2>&1
  fi
  if [ ! -f $r/test_ket_qua.json ]; then
    $PY eval_seg.py --run $r --luu_anh > runs_${bt}_s${s}_eval.log 2>&1
  fi
  echo "xong $bt seed $s"
done

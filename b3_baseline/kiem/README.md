# Tự kiểm baseline B3

Kết quả đã chạy 23/11/2026: [`ket_qua.txt`](ket_qua.txt).

```
python kiem/tinh_lai_miou.py          # chỉ cần repo này + numpy
python kiem/kiem_md5.py --fmb <FMB> --ckpt <thư mục checkpoint>   # chạy trong b3_baseline/
```

- **Số chính: `mIoU_co_mat`** – bỏ pixel có nhãn thật 0, trung bình 13 lớp có mặt trong test; pixel có nhãn thật 1–14 bị đoán thành 0 tính là FN. Các số 7/14/15 lớp chỉ để đối chiếu với bài khác. So phương pháp khác thì phải chấm bằng cùng quy tắc.
- `tinh_lai_miou.py` viết độc lập với `code/chung.py`, đọc `runs/<run>/test_nham_last.npz`. Đây là kiểm lại **phép tính**, không phải train lại độc lập.
- **Checkpoint** (12 `last.pt`, 2,2 GB) không nằm trong Git: dataset Kaggle riêng tư `thanhmn123/sage-b3-fmb-checkpoints`, xin quyền Mân.
- **Giới hạn:** md5 chỉ bắt ảnh trùng hệt. 7 ảnh `leak` được chốt bằng mắt, không khẳng định 273 ảnh còn lại sạch tuyệt đối. Test đã được chấm nhiều lần, không phải test chưa từng dùng.
- Checkpoint đã học đủ 1.220 ảnh train: **không dùng dự đoán trên chính các ảnh đó** để dạy mô hình khác (router…). Chưa có dự đoán out-of-fold.

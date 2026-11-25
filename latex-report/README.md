# Báo cáo LaTeX

Báo cáo cuối kỳ về hệ thống MLOps cho bài toán phân loại trạng thái hỏng máy trên AI4I 2020. Bản PDF 50 trang được lưu tại [`../output/pdf/bao_cao_MLOps_bao_tri_du_doan.pdf`](../output/pdf/bao_cao_MLOps_bao_tri_du_doan.pdf).

## Cấu trúc

- `main.tex`: trang bìa, tóm tắt, định dạng và mục lục.
- `chapters/`: tám chương nội dung.
- `references.tex`, `appendices.tex`: nguồn tham khảo và phụ lục.
- `generate_assets.py`: kiểm tra dữ liệu/hiện vật rồi tạo bảng và hình từ kết quả chạy.
- `generated/`, `figures/`: bảng và hình có thể tạo lại; được lưu trong Git để biên dịch báo cáo không phải huấn luyện lại.

## Biên dịch

Sau khi đã chạy pipeline và sinh `reports/evidence.json`, đứng ở thư mục gốc repository:

```powershell
.\latex-report\build.ps1
```

Script chạy chương trình tạo hình, biên dịch hai lượt với XeLaTeX và sao chép PDF vào `output/pdf/`. Cần cài MiKTeX/XeLaTeX và Times New Roman trên máy. Nếu chỉ muốn biên dịch lại từ bảng và hình đã lưu trong Git, dùng `xelatex` hai lượt trong thư mục `latex-report` mà không chạy `generate_assets.py`.

Nguồn số liệu chính: `data/raw/ai4i2020.csv` (SHA-256 ghi trong báo cáo), `artifacts/metrics.json`, `artifacts/model.joblib` và `reports/evidence.json`. Đổi kết quả thực nghiệm thì chạy lại chương trình tạo bảng/hình trước khi biên dịch.

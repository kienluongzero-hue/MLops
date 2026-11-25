# Predictive Maintenance MLOps

Dự đoán khả năng máy công nghiệp bị hỏng từ thông số vận hành, đồng thời xây dựng một quy trình có thể chạy lại từ dữ liệu đến API dự đoán.

## Đề xuất đề tài

**A. Mục tiêu.** Xây dựng pipeline tải và kiểm tra dữ liệu, huấn luyện và so sánh mô hình, lưu kết quả thí nghiệm, đóng gói mô hình tốt nhất, rồi cung cấp API dự đoán. Đây là bài tập MLOps; dữ liệu mô phỏng không đủ để triển khai trong nhà máy thật.

**B. Framework và cách tích hợp.** [MLflow](https://mlflow.org/docs/latest/tracking/) ghi tham số và chỉ số đánh giá của từng lần huấn luyện, đồng thời lưu artifact của mô hình được chọn. [DVC](https://doc.dvc.org/user-guide) định nghĩa hai bước `download` và `train` trong `dvc.yaml`, theo dõi phiên bản dữ liệu và sản phẩm của pipeline. FastAPI phục vụ dự đoán; Docker đóng gói API.

**C. Dữ liệu ban đầu.** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i) của UCI, gồm 10.000 bản ghi mô phỏng. Mỗi bản ghi có loại sản phẩm, nhiệt độ không khí và quy trình, tốc độ quay, mô-men xoắn, độ mòn dụng cụ. Nhãn là `Machine failure`. Dữ liệu được tải từ UCI bằng script, không lưu bản sao trực tiếp trong Git. Nguồn: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C), giấy phép CC BY 4.0.

**D. Mô hình dự kiến.** Logistic Regression (mốc so sánh), Random Forest và XGBoost. Mô hình được chọn bằng average precision trên tập validation. Tập test chỉ dùng một lần để báo cáo cuối cùng.

**Môi trường:** Python 3.11–3.13, Docker, chạy trên máy cá nhân; không cần GPU hay dịch vụ đám mây.

## Thiết kế

```text
UCI CSV → kiểm tra dữ liệu → chia train/validation/test → tiền xử lý + huấn luyện
                                               ↓
                              MLflow: tham số, chỉ số, mô hình
                                               ↓
                              model.joblib + metrics.json → FastAPI
                              reference.csv + dữ liệu mới → báo cáo giám sát
```

Chỉ sáu đặc trưng cảm biến và loại sản phẩm được dùng để dự đoán. `UID`, `Product ID` là định danh; `TWF`, `HDF`, `PWF`, `OSF`, `RNF` là các nhãn nguyên nhân hỏng và không được dùng làm đặc trưng vì làm rò rỉ đáp án. Dữ liệu không có mốc thời gian thực tế để đánh giá dự báo trước theo thời gian; phép chia stratified được dùng cho bài toán phân loại bản ghi hiện tại.

## Kết quả chạy thử

Với seed 42, tỷ lệ hỏng là 3,39%. Chia dữ liệu thành 6.400 bản ghi train, 1.600 validation và 2.000 test. Kết quả PR-AUC trên validation: Logistic Regression **0,420**, Random Forest **0,712**, XGBoost **0,780**. XGBoost được chọn với ngưỡng cảnh báo 0,15; trên test đạt PR-AUC **0,825**, precision **0,614**, recall **0,794** (54 lỗi phát hiện, 14 lỗi bỏ sót, 34 cảnh báo nhầm). Số liệu đầy đủ được tạo lại ở `artifacts/metrics.json`.

## Chạy trên máy cá nhân

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[train,dev]"
python -m predictive_maintenance.data
python -m predictive_maintenance.train
python -m pytest
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Mở MLflow tại `http://127.0.0.1:5000`. Mô hình được chọn lưu ở `artifacts/model.joblib`, kết quả ở `artifacts/metrics.json`. Để chạy bằng DVC trong môi trường ảo đã kích hoạt: `dvc repro`. `dvc.lock` được đưa vào Git; bộ nhớ đệm DVC hiện ở máy cá nhân và chưa cấu hình remote. Khi clone repository, bước `download` sẽ lấy lại dữ liệu gốc từ UCI.

```powershell
uvicorn predictive_maintenance.api:app --host 127.0.0.1 --port 8000
```

Gọi API mẫu:

```powershell
$body = @{type='L'; air_temperature=298.1; process_temperature=308.6; rotational_speed=1551; torque=42.8; tool_wear=0} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $body
```

Chạy API bằng Docker sau khi đã huấn luyện:

Image chỉ cài các thư viện phục vụ API và dùng bản [XGBoost chỉ chạy CPU](https://xgboost.readthedocs.io/en/latest/install.html#minimal-installation-cpu-only). Các phiên bản trong `requirements-api.txt` khớp môi trường đã tạo `model.joblib`.

```powershell
docker build -t predictive-maintenance-api .
docker run --rm -p 8000:8000 -v "${PWD}/artifacts:/app/artifacts:ro" predictive-maintenance-api
```

## Giám sát dữ liệu đầu vào

Bước huấn luyện lưu đúng sáu đặc trưng của **tập train** vào `artifacts/reference.csv`. Công cụ giám sát nhận một CSV dữ liệu mới có cùng sáu cột, so sánh từng đặc trưng với tập tham chiếu rồi xuất báo cáo JSON. Với đặc trưng số, cảnh báo khi thống kê Kolmogorov–Smirnov ≥ 0,20 và p-value < 0,01; với `Type`, cảnh báo khi khoảng cách total variation ≥ 0,10. Báo cáo cũng ghi số giá trị thiếu hoặc không hợp lệ. Đây là các ngưỡng demo, cần hiệu chỉnh nếu có dữ liệu vận hành thật.

Tạo hai lô demo từ phần dữ liệu đã giữ lại sau khi chia train/test. Lô thứ hai được cố ý tăng nhiệt độ, mô-men xoắn và thay đổi tỷ lệ loại sản phẩm:

```powershell
python -m predictive_maintenance.simulate --output data/monitoring/normal.csv
python -m predictive_maintenance.monitor --current data/monitoring/normal.csv --output artifacts/monitoring/normal.json
python -m predictive_maintenance.simulate --drift --output data/monitoring/drifted.csv
python -m predictive_maintenance.monitor --current data/monitoring/drifted.csv --output artifacts/monitoring/drifted.json
```

Để giám sát dữ liệu mới, thay `--current` bằng đường dẫn CSV thực tế. Công cụ này chỉ phát hiện thay đổi phân phối đầu vào; nó chưa đo được chất lượng dự đoán khi chưa có nhãn thật và không tự động huấn luyện lại mô hình. Dữ liệu demo không được dùng để chọn mô hình hoặc điều chỉnh ngưỡng đánh giá trên tập test.

Khi chạy thử với 500 bản ghi mỗi lô, `normal.json` có trạng thái `ok`; `drifted.json` có trạng thái `alert` ở `Type`, nhiệt độ không khí và mô-men xoắn, đúng ba đặc trưng được sửa trong lô demo.

## Đánh giá và giới hạn

Nhóm báo cáo prevalence của lỗi, precision, recall, F1, average precision (PR-AUC) và ROC-AUC trên tập test. Ngưỡng cảnh báo được chọn trên tập validation để tối đa F2, ưu tiên phát hiện lỗi. Kết quả trên dữ liệu mô phỏng có thể cao hơn nhiều so với dữ liệu vận hành thật. API trả xác suất cho trạng thái tại bản ghi đầu vào, không dự báo thời gian còn lại trước khi máy hỏng.

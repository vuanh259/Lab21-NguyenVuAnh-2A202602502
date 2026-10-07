# Lab 21 — LoRA fine-tuning và đánh giá đối chứng

**Học viên:** Nguyễn Vũ Anh  
**Mã học viên:** 2A202602502  
**Lớp:** Track 3: AI Application

## Phạm vi và lựa chọn

Thí nghiệm sử dụng `unsloth/Qwen3.5-4B` theo cấu hình T4 của bộ lab, với tác vụ phân loại ticket chăm sóc khách hàng tiếng Việt thành bốn trường JSON: intent, urgency, product, sentiment. Tác vụ này cho phép đo trực tiếp độ đúng của từng trường và tách lỗi định dạng khỏi lỗi phân loại. Dùng cùng base model và dữ liệu cho baseline và tất cả adapter để so sánh có ý nghĩa.

Môi trường thực thi: Google Colab, Tesla T4 14,6 GB, fp16. Mã nguồn của lab ở commit `d27c1c0`. Tập dữ liệu gồm 250 mẫu; chia 225 train/25 validation với seed 42. Đánh giá đầy đủ trên 50 ticket mục tiêu và 15 câu hỏi regression. Không đặt `EVAL_LIMIT`, không giảm epoch mặc định.

## Mask và độ dài chuỗi

NB1 xác nhận assistant-only loss: ví dụ kiểm tra có 39/94 token được giám sát, tỷ lệ 0,4149; `answer_is_supervised=true`, `question_is_masked=true`. Trên tập train, 9014/20951 token được giám sát (43,0%). Vì vậy loss tập trung vào câu trả lời thay vì học lặp lại câu hỏi.

Phân bố 250 mẫu: mean 93,1; p50=93; p95=98; p99=100; max=101 token. Chọn `max_length=256`, theo gợi ý thực đo của NB1 và đủ chứa mọi mẫu. Lần đo dùng giới hạn 256 cho cả bốn cấu hình. Notebook tái lập áp dụng LAB_MAX_LENGTH=256 bằng dataclasses.replace trên TIER riêng của từng stage, giữ cấu hình phần cứng mặc định 1024 của bộ lab. Các tham số thực đo nằm trong runs.csv và length_choice.json. NB1 kiểm tra chat template giữ phần reasoning.

## Baseline đóng băng trước huấn luyện

| Cấu hình | Target field accuracy | Regression keyword recall | JSON format | Latency ms/mẫu |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0,000 | 0,791111 | 0,000 | 3173,836 |
| (b) Base + optimized prompt | 0,765 | 0,791111 | 1,000 | 1036,796 |

Baseline (b) tốt hơn (a) trước khi fine-tuning. Các số liệu được lưu trong `results/baselines_frozen.json`; SHA của optimized prompt là `719e74d3b6232053`. Target là trung bình độ đúng của bốn trường, không phải tỷ lệ ticket đúng toàn bộ. Regression đo keyword recall, không khẳng định đánh giá đầy đủ khả năng tổng quát của model.

## Huấn luyện và đối chứng

Dùng cùng seed 42, dữ liệu train 225 mẫu, assistant-only mask, max_length=256, batch=1 và gradient_accumulation=16 (effective batch 16, dưới 32). Cấu hình chuẩn có all text-linear, rank 16, alpha 32, LR=1e-4, bằng 10 lần mức full-FT 1e-5 dùng trong đối chứng. Scheduler cosine, warmup 3 bước, gradient checkpointing. Ngân sách mặc định là 2 epoch, 30 bước theo scheduler. Attention-only dùng q/v, tăng rank lên 283 và alpha 566 để khớp số tham số; chênh 8192 tham số, khoảng 0,02523%, dưới 5%. QLoRA dùng base 4-bit cả khi train và khi chấm adapter.

| Run | Placement | r / alpha | LR | Base | Trainable params | Train_loss trung bình | Train giây | VRAM GB | Bước |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| correct | text-linear | 16 / 32 | 0.0001 | fp16 | 32464896 | 0.6258 | 394.7 | 8.78 | 30 |
| attn_only | attn-only | 283 / 566 | 0.0001 | fp16 | 32456704 | 0.5386 | 255.9 | 8.79 | 30 |
| wrong_lr | text-linear | 16 / 32 | 1e-05 | fp16 | 32464896 | 1.5702 | 388.5 | 8.78 | 30 |
| qlora | text-linear | 16 / 32 | 0.0001 | 4-bit | 32464896 | 0.7058 | 466.5 | 3.86 | 30 |

Cột final_loss trong runs.csv do bộ lab ghi train_loss tổng hợp trung bình, không phải loss của riêng batch cuối. Toàn bộ pipeline NB1–NB5 mất khoảng 3125 giây (52,1 phút), gồm cả tải model, nạp trọng số và đánh giá. Log có một số grad_norm=NaN ở fp16; kiểm tra toàn bộ bốn file adapter cho thấy không có NaN/Inf và các tham số đều khác 0 (results/adapter_finite_check.json).

## Bốn nhóm đánh giá và giải phẫu đối chứng

Baseline (b) dùng OPTIMIZED_PROMPT đã chốt; FT dùng NAIVE_PROMPT ngắn theo thiết kế NB5. Các tập đánh giá và SHA prompt vẫn khớp bản đóng băng. Target là độ đúng trung bình của 4 trường. Format là tỷ lệ khóa bắt buộc hiện diện qua bộ parse JSON linh hoạt của lab; điểm 1,0 không tự chứng minh chỉ có JSON hoặc mọi giá trị đúng vocabulary. Regression dùng keyword recall, chuẩn hóa tiếng Việt giống nhau ở prediction và keyword. Latency dùng kết quả ms/mẫu của cùng hàm chấm.

| Run | Target | Regression | Format | Latency ms/mẫu | n target |
|---|---:|---:|---:|---:|---:|
| (a) base + naive prompt | 0 | 0.7911 | 0 | 3173.8 | 50 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1 | 1036.8 | 50 |
| (c) LoRA fine-tune | 0.98 | 0.5222 | 1 | 1352.4 | 50 |

| Run | Target | Format | Latency ms/mẫu | n |
|---|---:|---:|---:|---:|
| correct | 0.98 | 1 | 1352.4 | 50 |
| attn_only | 0.965 | 1 | 885.9 | 50 |
| wrong_lr | 0 | 0 | 5170.1 | 50 |
| qlora | 0.94 | 1 | 1761.1 | 50 |

Nhóm regression được đo cho (a), (b) và correct để tính gate bốn nhóm. Theo NB5, ba đối chứng được chấm target/format/latency; không có số regression của các đối chứng trong bảng này. Xếp hạng theo target: correct > attn_only > qlora > wrong_lr. Attention-only có train_loss thấp hơn correct nhưng target thấp hơn 1,5 điểm phần trăm; thứ tự theo loss khác thứ tự theo tác vụ. Wrong_lr không tạo được JSON đủ khóa trong đánh giá này nên target và format đều bằng 0. QLoRA giảm VRAM khoảng 56% nhưng target thấp hơn 4 điểm phần trăm và latency cao hơn correct khoảng 30%; đây là đánh đổi thực đo của môi trường này. Chỉ số valid_trace_rate của correct là 0,0. Tác vụ train chủ yếu là JSON; chỉ số này riêng nó chưa đủ đánh giá năng lực suy luận tổng quát. Kiểm tra template ở NB1 là kiểm tra giữ cấu trúc tag, không phải chứng minh chất lượng reasoning.

## Ví dụ định tính: cả thắng, sai và thua

Đối chiếu đầy đủ 50 ticket: 34 ca FT thắng, 16 ca hòa, 0 ca thua baseline (b) theo field accuracy. FT còn 4 ticket sai urgency (i=3,12,41,46); 46/50 ticket đúng cả bốn trường, trong khi target=0,98 là 196/200 trường đúng. Ba ca tệ nhất NB5 (i=3,12,41) đều được đưa dưới đây. Vì không có ca thua baseline trong nhóm target, hai ca thua thật được thể hiện ở regression. Trong 15 câu regression có 1 ca thắng, 7 ca thua và 7 ca hòa. Không đổi dữ liệu hay điểm số để tạo ca thua.

Ví dụ baseline được tái sinh bổ sung với đúng model và prompt đã đóng băng để có dự đoán từng mẫu; điểm tái sinh target=0,765 và regression=0,791111 khớp baseline gốc. Các aggregate baseline gốc và verdict gốc được giữ nguyên.

### Target i=0

**Ticket:** Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.

**Nhãn:** `{"intent":"doi_tra","urgency":"cao","product":"chuột không dây","sentiment":"tich_cuc"}`

**Fine-tune:** `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}`

**Baseline (b):** `{"intent": "hoan_tien", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}`

Baseline nhầm intent hoan_tien; nhãn và FT là doi_tra. Các trường còn lại đúng. Đây là ca FT thắng 1,00 so với 0,75.

### Target i=1

**Ticket:** Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình.

**Nhãn:** `{"intent":"hoan_tien","urgency":"trung_binh","product":"ốp lưng điện thoại","sentiment":"tieu_cuc"}`

**Fine-tune:** `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}`

**Baseline (b):** `{"intent": "hoan_tien", "urgency": "cao", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}`

Baseline dự đoán urgency=cao cho “Sớm nhé”; nhãn và FT là trung_binh. FT thắng 1,00 so với 0,75.

### Target i=3

**Ticket:** Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều.

**Nhãn:** `{"intent":"hoan_tien","urgency":"thap","product":"bình giữ nhiệt","sentiment":"tich_cuc"}`

**Fine-tune:** `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}`

**Baseline (b):** `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}`

“Khi nào tiện” có nhãn urgency=thap, nhưng cả hai mô hình dự đoán trung_binh. FT sai một trường và hòa baseline ở 0,75.

### Target i=12

**Ticket:** Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều.

**Nhãn:** `{"intent":"san_pham_loi","urgency":"thap","product":"áo khoác gió","sentiment":"tich_cuc"}`

**Fine-tune:** `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}`

**Baseline (b):** `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}`

Ticket sản phẩm lỗi kèm “Khi nào tiện” vẫn bị gán urgency=trung_binh. Nhãn là thap; cả FT và baseline đạt 0,75.

### Target i=41

**Ticket:** Cho mình hỏi, mình đặt đèn bàn LED mã đơn OD436045. Giao hàng chậm. Khi nào tiện. Cảm ơn shop nhiều.

**Nhãn:** `{"intent":"van_chuyen","urgency":"thap","product":"đèn bàn LED","sentiment":"tich_cuc"}`

**Fine-tune:** `{"intent": "van_chuyen", "urgency": "trung_binh", "product": "đèn bàn LED", "sentiment": "tich_cuc"}`

**Baseline (b):** `{"intent": "hoi_thong_tin", "urgency": "trung_binh", "product": "đèn bàn LED", "sentiment": "tich_cuc"}`

FT sửa được intent từ hoi_thong_tin của baseline thành van_chuyen, nhưng urgency vẫn là trung_binh thay vì thap. FT chưa đúng hết: 0,75 so với baseline 0,50.

### Regression i=2 — FT thua

**Câu hỏi:** 1 km bằng bao nhiêu mét?

**Keyword đúng:** `1000`. Baseline keyword recall=1; FT=0.

**Fine-tune:** `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}`

**Baseline (b), trích đoạn:** “Kết quả: 1 km = 1000 m.”

FT áp dụng schema phân loại ticket lên câu hỏi kiến thức thông thường, không trả lời giá trị cần thiết. Đây là lỗi hành vi trên regression, dù JSON có vẻ hợp lệ. Dự đoán đầy đủ của cả hai được giữ trong results/predictions_regression_*.json.

### Regression i=9 — FT thua

**Câu hỏi:** Một năm có bao nhiêu tháng?

**Keyword đúng:** `12`. Baseline keyword recall=1; FT=0.

**Fine-tune:** `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}`

**Baseline (b), trích đoạn:** “Một năm bình thường có 12 tháng.”

FT áp dụng schema phân loại ticket lên câu hỏi kiến thức thông thường, không trả lời giá trị cần thiết. Đây là lỗi hành vi trên regression, dù JSON có vẻ hợp lệ. Dự đoán đầy đủ của cả hai được giữ trong results/predictions_regression_*.json.

## Phân tích verdict (dựa trên NB5 đã đo)

Verdict là FAIL vì regression giảm 0,268889, vượt xa mức giảm cho phép 0,02, dù target tăng 0,215 so với baseline tối ưu. Cấu hình chuẩn đạt 0,98 về độ đúng trung bình của bốn trường; đây là cải thiện rõ trên tập 50 ticket này. Tuy nhiên, kết luận triển khai phải xét cả hai điều kiện của gate. Chỉ so với baseline ngây thơ có target bằng 0 sẽ che mất yêu cầu quan trọng: đối thủ thực sự là baseline đã tối ưu trước huấn luyện. Điểm format bằng 1,0 cũng không bù được việc trả lời kém hơn trên các câu hỏi regression. Latency của fine-tune khoảng 1352,4 ms/mẫu, tăng khoảng 30,4% so với 1036,8 ms/mẫu của baseline (b), dù dùng prompt ngắn hơn. Kết quả này chưa đủ lý do thay baseline bằng adapter chuẩn cho một trợ lý cần giữ năng lực rộng. Một bước tiếp theo có thể thử là thêm 1–5% replay dữ liệu tổng quát và đánh giá lại bằng gate giữ nguyên; thí nghiệm replay chưa được thực hiện trong kết quả core này. Regression ở đây là keyword recall trên 15 câu hỏi, nên diễn giải là suy giảm trên bộ kiểm tra này, chưa phải phép đo toàn diện mọi năng lực của model.

## Bài học và giới hạn rút ra

Bài học đầu tiên là cần chốt baseline và tập đánh giá trước khi nhìn kết quả fine-tuning. Prompt tối ưu đã nâng target từ 0 lên 0,765 mà chưa huấn luyện, cho thấy prompt engineering là một đối thủ đáng kể. Bài học thứ hai là phải kiểm chứng loss mask và độ dài chuỗi bằng dữ liệu. Assistant-only mask được kiểm tra trực tiếp, còn giới hạn 256 token được chọn sau khi đo p95=98 và max=101, thay vì dùng một số lớn theo thói quen. Bài học thứ ba là loss huấn luyện không quyết định chất lượng ngoài tập train: attention-only có train_loss trung bình 0,5386, thấp hơn 0,6258 của cấu hình chuẩn, nhưng target lại thấp hơn (0,965 so với 0,98). Vì thế, xếp hạng bằng loss sẽ cho thứ tự khác xếp hạng bằng tác vụ. Bài học thứ tư là tiết kiệm VRAM chưa đồng nghĩa tiết kiệm thời gian: QLoRA dùng 3,86 GB nhưng thời gian train dài hơn cấu hình fp16 trong môi trường này. Kết quả chỉ đại diện một seed, một tập dữ liệu nhỏ và một GPU. Chưa có nhiều lần chạy để ước lượng độ biến thiên hoặc khoảng tin cậy. Log fp16 có gradient NaN xen kẽ; các trọng số đã lưu đều hữu hạn, nhưng chưa ghi số cập nhật thực bị GradScaler bỏ qua. Do đó, bảng 30 bước thể hiện ngân sách bước theo scheduler của bộ lab, chưa chứng minh mọi cấu hình có đúng 30 cập nhật hữu hiệu. Đây là giới hạn phải giữ khi diễn giải chênh lệch nhỏ giữa các cấu hình.


## Tái lập và các file bằng chứng

Mã nguồn: [VinUni Lab21](https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-Lab), commit d27c1c02ebe99f32f52f706be88b4c30fb1d7fca. [Notebook thực thi](https://colab.research.google.com/drive/1atokmxT0U-Qd0LXFAm7rIXAphLw4fSdm) dùng T4 và full eval. Python 3.13.15; torch 2.11.0+cu130; transformers 5.18.0; TRL 1.14.2. Danh sách phiên bản đầy đủ trong results/environment.json.

Chạy các ô từ trên xuống trong notebook đã lưu. Ô pipeline đo token_stats khi cần, đặt LAB_MAX_LENGTH=256 theo dữ liệu, áp dụng giới hạn riêng trên TIER, xuất dự đoán và tạo lại colab/*.ipynb từ notebooks/*.py để đồng bộ. Khi tái chạy từ archive bằng terminal, đặt COMPUTE_TIER=T4 và LAB_MAX_LENGTH=256; bỏ EVAL_LIMIT và EPOCHS để dùng full eval và mặc định 2 epoch; chạy python scripts/colab_run.py nb1 nb2 nb3 nb4 nb5, rồi python scripts/verify.py.

Bằng chứng chính: template_check.json, mask_proof.json, token_stats.json, length_choice.json, baselines_frozen.json, runs.csv, verdict.json, autopsy.json, qualitative.json, predictions_correct.json, predictions_baseline_b.json và các predictions_regression_*.json. Log kiểm tra nộp bài cuối lưu ở results/verification_log.txt. Gói nộp gồm report, results, adapter correct, notebook thực thi và mã nguồn/notebook cần để tái lập.

Report ghi nhận lần chạy này. Khi huấn luyện lại, cập nhật bảng số và ví dụ từ kết quả mới trước khi đóng gói; ô final kiểm tra hash các dữ liệu đầu vào của report để ngăn dùng nhầm report cũ.


## Bonus B5 — GitHub và HuggingFace Hub

Adapter LoRA `correct` được công bố công khai trên [HuggingFace Hub](https://huggingface.co/vustaz/Lab21-Qwen3.5-4B-LoRA). Mã nguồn, notebook, report và kết quả nằm tại [GitHub](https://github.com/vuanh259/Lab21-LoRA-2A202602502). Đây là adapter PEFT, cần nạp cùng base model `unsloth/Qwen3.5-4B`, không phải model đã merge. Các link được tổng hợp trong `LINKS.md`. Phần B5 không thay đổi điểm đánh giá hoặc verdict FAIL do regression; chưa thực hiện B1–B4.

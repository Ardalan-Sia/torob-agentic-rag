# Torob Product Retrieval RAG — Phase 1 Report

## 1) Goal
هدف فاز ۱: ساخت یک سیستم بازیابی محصول (Product Retrieval) با پایپ‌لاین قابل بازتولید شامل:
- EDA + Cleaning
- Feature/Embedding extraction (text/image)
- ذخیره‌سازی embeddingها در Vector DB (FAISS local)
- سه baseline: Text-only, Image-only, Multimodal
- Evaluation و تحلیل خطا

دیتا: dump از ترب شامل جداول relational (base_products, members, searches, base_views, final_clicks, …)

---

## 2) Dataset Overview (Sanity)
Checkpoint 01 (Load + FK sanity):

- base_products: rows=1,022,298 cols=8 missing_rate≈0.0004
- members: rows=1,948,665 cols=4 missing_rate≈0
- searches: rows=588,347 cols=9
- base_views: rows=199,916 cols=4
- final_clicks: rows=17,371 cols=4

Consistency:
- missing FK (members→base, members→shop, base→category, base→brand): 0
- invalid prices: price<=0 count=2
- extreme price max: 1.99e17

---

## 3) EDA Findings
Checkpoint 02:

- price max: 1.99e17 (outlier شدید)
- price p99.9: 380,000,000
- count(price<=0): 2
- members_per_base max: 1548
- search query length p95: 28

نتیجه: نیاز به پاکسازی قیمت‌ها و cap کردن outliers برای جلوگیری از اثرگذاری شدید روی ranking/analytics.

---

## 4) Cleaning & Unification
Checkpoint 03:

- members_before: 1,948,665
- members_invalid_price (dropped): 440
- members_after: 1,948,225
- price_min_valid: 1
- price_max_cap: 1,000,000,000
- bp_rows: 1,022,298
- bp_text_empty: 0

Outputs:
- artifacts/clean/base_products_clean.parquet
- artifacts/clean/members_clean.parquet
- artifacts/clean/unified_base_products.parquet

---

## 5) Subset Strategy (CPU/Scale constraint)
هدف: ساخت subset قابل دفاع برای embedding روی CPU.

Subset v2 (logs-weighted) ساخته شد:
- max_base_products: 200,000
- روش: امتیازدهی وزن‌دار با logs (impressions + views + final_clicks)
  - impression weight=1
  - view weight=5
  - final_click weight=20

Output:
- artifacts/subset/base_products_subset_v2.parquet

Observation (مهم):
پوشش بعضی intentها (مثل “گوشی سامسونگ S23”) در subset پایین بود و باعث خروجی‌های نامرتبط در demo شد. این محدودیت به داده/پوشش subset مربوط است نه صرفاً FAISS.

---

## 6) Text-only Retrieval Baseline
Embedding:
- model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- n_items: 200,000
- dims: 384
- normalize: True
FAISS:
- IndexFlatIP, ntotal=200,000

Evaluation (Checkpoint 06) با ground-truth از base_views (clicked base products):
- eval_samples: 5000 (از pool شامل searchهایی که حداقل یک clicked item داخل subset بودند)
- recall@1: 0.0062
- recall@5: 0.0216
- recall@10: 0.0328
- mrr@10: 0.01254
- ndcg@10: 0.01406
- failures: 4836 / 5000

Error Analysis (Checkpoint 07A):
نمونه‌ها نشان می‌دهد:
- queryهای کوتاه/کلی (مثلاً “لیوان”) ambiguity بالا دارند
- برخی queryهای ترکیبی (مثلاً “جا ادویه کوهنوردی”) drift به سمت کلمات غالب رخ می‌دهد

Artifacts:
- artifacts/eval/text/eval_summary.json
- artifacts/eval/text/failure_examples_readable.csv

---

## 7) Image-only Retrieval Baseline (CPU)
Subset تصویری:
- top 20,000 base products بر اساس view_count و image_url موجود

Download:
- attempted: 20,000
- download_ok: 16,236
- download_fail: 3,764

Image Embedding:
- model: openai/clip-vit-base-patch32
- n_items: 16,236
- dims: 512
- normalize: True
- embeddings.npy size: ~32MB

FAISS:
- IndexFlatIP, ntotal=16,236

Sanity Eval (Checkpoint 08F):
- self_recall@1: 0.993964 (انتظار≈1، اختلاف جزئی به دلیل تکراری/نویز یا خطای جزئی index mapping)
- top1 score ~1.0 (به علت cosine/IP با خود بردار)

Artifacts:
- artifacts/eval/image/clip_v1/sanity_report.json

---

## 8) Multimodal Retrieval Baseline (Text + Image) — Late Fusion
روش: Late Fusion روی دو بازیاب مستقل (text FAISS + image FAISS) و ترکیب امتیازها:
score = alpha * text_score + (1-alpha) * image_score

Corpus مشترک:
- n_items: 16,236 (فقط محصولاتی که تصویر دانلود/embedding شده)

Text embeddings برای corpus مشترک:
- model: paraphrase-multilingual-MiniLM-L12-v2
- n_items: 16,236
- dims: 384
- FAISS text ntotal=16,236

Sanity Eval (Checkpoint 09E):
- sample_n: 2000
- recall@1/5/10 = 1.0
- mrr@1/5/10 = 1.0
توضیح: این sanity است چون query برابر embedding خود آیتم است (نه لاگ واقعی multimodal).

Observation:
در demo با متن “گاز کولر R22”، text component بعضی نتایج را به سمت کالاهای گازسوز/کولر گازی/کمپرسور برد. برای بهبود، یا باید alpha را کاهش داد (image-dominant) یا text encoder retrieval-tuned شود (مثلاً E5) یا gating/keyword boost اضافه شود (فاز ۲).

Artifacts:
- artifacts/eval/multimodal/clip_v1/mm_sanity_eval.json

---

## 9) Reproducibility & Engineering Notes
- کلیه مسیرها و پارامترها در config.yaml نگهداری شد.
- خروجی‌های checkpoint در artifacts ذخیره و قابل بازتولید هستند.
- پیشنهاد برای فاز ۲:
  - versioning بهتر embedding runs
  - اضافه کردن lockfile (requirements.txt / uv / poetry)
  - seed ثابت در sampling/eval

---

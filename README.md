# TagGenie 🧞

**Zero-Shot Directory Classifier Utility**

I run a niche directory for regulated service providers (Lawyers, Consultants, and Tax Advisors). TagGenie is a local AI utility I built to auto-categorize business listings, lead lists, and directory submissions. It uses a Zero-Shot Classification model (`facebook/bart-large-mnli`) to assign the best matching tag from a user-provided list, with a built-in "Safety Valve" for irrelevant inputs.

## 🚀 Features

* **Zero-Shot Learning:** No training required. Just provide your custom tags at runtime.
* **The "Safety Valve":** Automatically appends "None of the above" to your tag list. If the model is unsure, it defaults to "None" rather than forcing a wrong category.
* **Batch Processing:** Process thousands of rows in a CSV file with a progress bar.
* **Local & Private:** Runs 100% locally using Hugging Face Transformers. No API costs, no data leakage.

## 📦 Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Install dependencies
poetry install
```

## 🛠️ Usage

### 1. Single Text Classification
Great for testing or quick lookups.

```bash
# Syntax
poetry run python main.py classify "TEXT_TO_CLASSIFY" --tags "Tag1,Tag2,Tag3"

# Example
poetry run python main.py classify "We provide legal visa consulting services" --tags "Legal,Travel,Real Estate"
```

**Output:**
```text
Result: Legal
Top 3 Matches:
1. Legal (92.5%)
2. Travel (5.1%)
3. None of the above (1.2%)
```

### 2. Batch CSV Processing
Process an entire spreadsheet of leads.

```bash
# Syntax
poetry run python main.py process-csv \
    ./input.csv \
    ./output.csv \
    --column "Description" \
    --tags "SaaS,Agency,Ecommerce,Consulting"

# Example
poetry run python main.py process-csv ./leads_raw.csv ./leads_tagged.csv --column "Company Bio" --tags "Sustainable,Plastic-Free,Recycled"
```

**Output:**
Generates a new CSV (`output.csv`) with two appended columns:
* `Predicted_Tag`: The winner (or "None").
* `Confidence_Score`: The probability score (0.0 - 1.0).

## 🛡️ Compliance & Safety

**Intended Purpose (EU AI Act):**
This tool is intended to assist human operators in categorizing business data. It is **not** intended to be a fully autonomous decision-maker for critical business processes without human oversight.

**Risk Management:**
* **False Positives:** The model may confidently categorize incorrect data.
* **Human Oversight:** It is recommended to manually review any predictions with a `Confidence_Score` below 0.70.

## 🏗️ Technical Details

* **Model:** `facebook/bart-large-mnli` (Downloaded to local cache on first run).
* **Architecture:** PyTorch + Hugging Face Transformers.
* **Interface:** Typer (CLI) + Rich (UI).


<!-- SHOWCASE: false -->

# LLM Reliability Evaluation

> Benchmarks four large language models across math reasoning and factual accuracy tasks, measuring response accuracy, self-consistency, and calibration of self-reported confidence.

![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Language](https://img.shields.io/badge/language-Python-blue)
![Semester](https://img.shields.io/badge/semester-Spring%202026-orange)

---

## Course Information

| Field                  | Details                                                                                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Course Title           | Computer Understanding of Natural Language                                                                                                                                      |
| Course Number          | CAP 6640                                                                                                                                                                        |
| Semester               | Spring 2026                                                                                                                                                                     |
| Assignment Title       | Final Course Project                                                                                                                                                            |
| Assignment Description | Investigate the ability of large language models to handle NLP tasks by designing and executing a structured evaluation pipeline across multiple models and benchmark datasets. |

---

## Project Description

This project evaluates the reliability of four large language models (GPT-3.5, GPT-4, Claude Haiku, and Gemini 2.5 Flash Lite) on two established NLP benchmarks: GSM8k for grade-school math reasoning and TruthfulQA for factual multiple-choice accuracy. Each model is queried across multiple trials per question, and responses are parsed to extract answers, reasoning, and self-reported confidence scores. The analysis pipeline computes accuracy, self-consistency, semantic variance, and confidence calibration, then produces comparative figures across all models and datasets.

---

## Screenshots / Demo

> _No screenshot available. Add one with: `![Demo](docs/your-image.png)`_

---

## Results

When `query_models.py` completes, results are appended to the path specified by `--output` (default: `results/results.csv`). Running `analysis.py` against that CSV produces a metrics summary and a set of figures.

**Sample terminal output during querying:**

```
[LOADING] 50 questions from each dataset...
[LOADING] 100 total questions.
--- Running Haiku3 | 5 trial(s) | 100 questions ---
Haiku3 | GSM8k Q#1 | Answer: 42 | Confidence: 85
Haiku3 | GSM8k Q#2 | Answer: 137 | Confidence: 72
...
--- Haiku3 complete. Results appended to results/results.csv ---
[SAVED] Results saved to: results/results.csv
```

**Sample terminal output during analysis:**

```
[LOADING] Results from results/results.csv...
[LOADING] 2000 valid rows.
[SAVED] results/analysis/metrics.csv
[SAVED] results/analysis/accuracy.png
[SAVED] results/analysis/self_consistency.png
[SAVED] results/analysis/semantic_variance.png
[SAVED] results/analysis/confidence.png
```

**Interpreting `metrics.csv`:** Each row represents one model-dataset combination and includes four metrics:

| Metric            | Range     | Interpretation                                                                                |
| ----------------- | --------- | --------------------------------------------------------------------------------------------- |
| Accuracy          | 0.0 - 1.0 | Fraction of answers matching ground truth                                                     |
| Self-Consistency  | 0.0 - 1.0 | Frequency of the modal answer across trials; 1.0 means the model always gives the same answer |
| Semantic Variance | 0.0 - 1.0 | Mean cosine distance between trial responses; lower values indicate more consistent reasoning |
| Mean Confidence   | 0 - 100   | Average self-reported confidence score; compare against accuracy to assess calibration        |

Per-question accuracy and confidence bar charts are saved under `results/analysis/per_question_analysis/`. If accuracy and confidence diverge significantly for a question, the model may be overconfident or underconfident on that item. API or parse errors are recorded in `error_log.txt` in the project root.

---

## Key Concepts

`Large Language Models` `Benchmark Evaluation` `Self-Consistency` `Confidence Calibration` `Semantic Variance` `GSM8k` `TruthfulQA` `Prompt Engineering` `NLP Evaluation`

---

## Languages & Tools

- **Language:** Python 3.10+
- **APIs:** OpenAI API, Anthropic API, Google Gemini API
- **Key Libraries:** `datasets`, `sentence-transformers`, `pandas`, `matplotlib`, `scikit-learn`
- **Build System:** pip / requirements.txt

---

## File Structure

```
llm-reliability-evaluation/
├── config.env                  # API keys (not committed)
├── requirements.txt            # Third-party dependencies
├── error_log.txt               # Runtime API/parse error log (generated)
├── results/
│   ├── results.csv             # Raw query output (generated)
│   └── analysis/
│       ├── metrics.csv         # Aggregated metrics table (generated)
│       ├── accuracy.png        # Accuracy bar chart (generated)
│       ├── self_consistency.png
│       ├── semantic_variance.png
│       ├── confidence.png
│       └── per_question_analysis/  # Per-model, per-question figures (generated)
└── src/
├── data_loader.py          # Downloads and standardizes GSM8k and TruthfulQA samples
├── query_models.py         # Queries LLM APIs and logs results to CSV
└── analysis.py             # Computes metrics and generates comparison figures
```

---

## Installation & Usage

### Prerequisites

- Python 3.10+
- API keys for OpenAI, Anthropic, and Google Gemini

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/alexneilgreen/UCF-ComputerUnderstandingOfNaturalLanguage-LLMReliabilityEval.git
cd UCF-ComputerUnderstandingOfNaturalLanguage-LLMReliabilityEval

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create config.env in the project root
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here

# 4. (Optional) Pre-download datasets without querying any APIs
python src/query_models.py --download-only

# 5. Run queries
python src/query_models.py [--trials N] [--model MODEL] [--questions N] [--output PATH]

# 6. Run analysis
python src/analysis.py [--input PATH] [--output_dir PATH]
```

### Controls

| Argument          | Default               | Description                                                 |
| ----------------- | --------------------- | ----------------------------------------------------------- |
| `--trials`        | `5`                   | Number of times each question is sent to the model          |
| `--model`         | `all`                 | Model to query: `all`, `GPT3.5`, `GPT4`, `Haiku3`, `Gemini` |
| `--questions`     | `50`                  | Number of questions sampled per dataset                     |
| `--output`        | `results/results.csv` | Output CSV path; appends if file already exists             |
| `--download-only` | N/A                   | Cache datasets and exit without querying APIs               |
| `--input`         | `results/results.csv` | Input CSV for analysis script                               |
| `--output_dir`    | `results/analysis`    | Directory for metrics CSV and figures                       |

---

## Academic Integrity

This repository is publicly available for **portfolio and reference purposes only**.
Please do not submit any part of this work as your own for academic coursework.

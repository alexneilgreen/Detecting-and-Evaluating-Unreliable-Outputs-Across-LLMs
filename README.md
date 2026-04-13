# LLM Reliability Evaluation

Queries GPT-3.5, GPT-4, Claude Haiku, and Gemini 2.5 Flash Lite with questions from GSM8k and TruthfulQA-multi, logging responses and self-reported confidence scores to a CSV for later analysis.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `config.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   GEMINI_API_KEY=your_key_here
   ```

## Running

```bash
python src/query_models.py [--trials N] [--model MODEL] [--questions N] [--output PATH]
```

## Arguments

| Argument          | Default               | Description                                                          |
| ----------------- | --------------------- | -------------------------------------------------------------------- |
| `--trials`        | `5`                   | Number of times each question is sent to the model                   |
| `--model`         | `all`                 | Model to query: `all`, `GPT3.5`, `GPT4`, `Haiku3`, `Gemini`          |
| `--questions`     | `50`                  | Number of questions sampled per dataset                              |
| `--output`        | `results/results.csv` | Output CSV path — appends if file already exists                     |
| `--download-only` | N/A                   | Download and cache both datasets then exit without querying any APIs |

## Examples

```bash
# Test API connection with 1 question, Haiku only
python src/query_models.py --model Haiku3 --questions 1 --output results/test.csv

# Full run, all models, 5 trials
python src/query_models.py --trials 5 --output results/full_run.csv

# Run only Gemini, save separately
python src/query_models.py --model Gemini --output results/gemini.csv
```

## Results

Results are saved to the path specified by `--output` (default: `results/results.csv`).

Columns: `AI Model | Question Number | Dataset | Question | Ground Truth Work | Ground Truth Answer | Response | Answer | Confidence`

API errors are logged to `error_log.txt` in the project root.

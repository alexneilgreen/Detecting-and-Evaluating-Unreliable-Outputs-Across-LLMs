"""
This file handles generating the queries from the datasets
and then prompting the LLM APIs.

The file also logs all responses and confidence scores into 
a CSV file at the defined output.
"""

import argparse
import csv
import os
import re
import sys
import time

from dotenv import load_dotenv
from data_loader import load_gsm8k, load_truthfulqa

load_dotenv("config.env")

# CSV Handling

CSV_COLUMNS = ["AI Model", "Question Number", "Dataset", "Question", "Response", "Confidence"]
ERROR_LOG_FILE = "error_log.txt"

def init_csv(output_path):
    """Write header row only if the file does not already exist."""
    if not os.path.exists(output_path):
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

def append_row(output_path, row):
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)

def log_error(message):
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# Prompt Handeling

GSM8K_SYSTEM = (
    "You are a math problem solver. "
    "Answer with the final numeric value only — no units, no working, no explanation. "
    "Use the following format exactly:\n"
    "ANSWER: <numeric value>\n"
    "CONFIDENCE: <integer 0-100, where 100 means you are certain your answer is correct and 0 means you have no confidence your answer is correct>"
)

TRUTHFULQA_SYSTEM = (
    "You are answering a multiple choice question. "
    "Respond with the letter of your chosen answer only — no explanation. "
    "Use the following format exactly:\n"
    "ANSWER: <letter>\n"
    "CONFIDENCE: <integer 0-100, where 100 means you are certain your answer is correct and 0 means you have no confidence your answer is correct>"
)

def build_prompt(question):
    """Returns (system_prompt, user_message) for a given question dict."""
    if question["dataset"] == "GSM8k":
        return GSM8K_SYSTEM, question["question"]
    else:
        return TRUTHFULQA_SYSTEM, question["question"]

# Response Parser

def parse_response(raw_text):
    """
    Parses ANSWER and CONFIDENCE from model response text.

    Returns (answer, confidence) — both as strings.
    Returns (raw_text, "PARSE_ERROR") if format is not followed.
    """
    answer_match = re.search(r"ANSWER:\s*(.+)", raw_text, re.IGNORECASE)
    confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", raw_text, re.IGNORECASE)

    answer = answer_match.group(1).strip() if answer_match else None
    confidence = confidence_match.group(1).strip() if confidence_match else None

    if answer is None or confidence is None:
        return raw_text.strip(), "PARSE_ERROR"

    return answer, confidence

# API Calls

def call_openai(model_name, system_prompt, user_message):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=1.0,
    )
    return response.choices[0].message.content.strip()

def call_claude(system_prompt, user_message):
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        temperature=1.0,
    )
    return response.content[0].text.strip()

def call_gemini(system_prompt, user_message):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        system_instruction=system_prompt,
    )
    response = model.generate_content(
        user_message,
        generation_config={"temperature": 1.0},
    )
    return response.text.strip()

# Model Handeling

MODEL_MAP = {
    "GPT3.5": ("openai", "gpt-3.5-turbo"),
    "GPT4":   ("openai", "gpt-4-turbo"),
    "Haiku3": ("claude", None),
    "Gemini": ("gemini", None),
}

def call_model(model_key, system_prompt, user_message):
    provider, model_name = MODEL_MAP[model_key]
    if provider == "openai":
        return call_openai(model_name, system_prompt, user_message)
    elif provider == "claude":
        return call_claude(system_prompt, user_message)
    elif provider == "gemini":
        return call_gemini(system_prompt, user_message)

# Calling APIs

def run_queries(questions, model_key, trials, output_path):
    """
    For each question, runs # of `trials` API calls and appends results to CSV.
    
    On failure: logs the error, writes ERROR rows to CSV, pauses for user input,
    then retries the same question before continuing.
    """
    for q in questions:
        system_prompt, user_message = build_prompt(q)

        for trial in range(1, trials + 1):
            success = False
            while not success:
                try:
                    raw = call_model(model_key, system_prompt, user_message)
                    answer, confidence = parse_response(raw)
                    append_row(output_path, {
                        "AI Model": model_key,
                        "Question Number": q["question_number"],
                        "Dataset": q["dataset"],
                        "Question": q["question"],
                        "Response": answer,
                        "Confidence": confidence,
                    })
                    success = True
                    # Small delay to respect rate limits
                    time.sleep(0.5)

                except Exception as e:
                    error_msg = (
                        f"ERROR | Model: {model_key} | Dataset: {q['dataset']} | "
                        f"Q#: {q['question_number']} | Trial: {trial} | {str(e)}"
                    )
                    print(f"\n[ERROR] {error_msg}")
                    log_error(error_msg)

                    append_row(output_path, {
                        "AI Model": model_key,
                        "Question Number": q["question_number"],
                        "Dataset": q["dataset"],
                        "Question": q["question"],
                        "Response": "API_ERROR",
                        "Confidence": "API_ERROR",
                    })

                    input("\nPress Enter to retry this question...")
                    print(f"Retrying Q#{q['question_number']} (Trial {trial})...")

# Main

def main():
    parser = argparse.ArgumentParser(
        description="Query LLMs with GSM8k and TruthfulQA questions and log results."
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of times each question is sent to the model (default: 1)."
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "GPT3.5", "GPT4", "Haiku3", "Gemini"],
        help="Which model to query (default: all)."
    )
    parser.add_argument(
        "--questions", type=int, default=50,
        help="Number of questions to sample from each dataset (default: 50)."
    )
    parser.add_argument(
        "--output", type=str, default="results/results.csv",
        help="Path to output CSV file. Appends if file already exists (default: results/results.csv)."
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Download and cache both datasets then exit without querying any APIs."
    )
    args = parser.parse_args()

    print(f"Loading {args.questions} questions from each dataset...")
    all_questions = load_gsm8k(args.questions) + load_truthfulqa(args.questions)
    print(f"Loaded {len(all_questions)} total questions.")

    if args.download_only:
        print("Datasets downloaded and cached. Exiting (--download-only).")
        sys.exit(0)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    init_csv(args.output)

    models_to_run = list(MODEL_MAP.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        print(f"\n--- Running {model_key} | {args.trials} trial(s) | {len(all_questions)} questions ---")
        run_queries(all_questions, model_key, args.trials, args.output)
        print(f"--- {model_key} complete. Results appended to {args.output} ---")

    print(f"\nAll done. Results saved to: {args.output}")

if __name__ == "__main__":
    main()
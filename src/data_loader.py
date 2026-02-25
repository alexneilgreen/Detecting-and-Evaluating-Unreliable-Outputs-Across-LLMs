"""
This file handles downloading GSM8k and TruthfulQA-multi datasets.

It returns a seeded random sample of the questions after their 
formats are standardized.
"""

import re
import random
from datasets import load_dataset

def load_gsm8k(num_questions, seed = 42):
    """
    Returns list of dicts:
        {
            "dataset": "GSM8k",
            "question_number": int,
            "question": str,
            "answer": str | int,          # numeric ground truth only
        }
    """
    dataset = load_dataset("gsm8k", "main", split="test")

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(num_questions, len(dataset)))

    results = []
    for counter, idx in enumerate(indices, start=1):
        row = dataset[idx]
        raw_answer = row["answer"]

        # Parse the numeric answer from the '#### 42' format
        match = re.search(r"####\s*(-?[\d,]+)", raw_answer)
        ground_truth = match.group(1).replace(",", "") if match else raw_answer.strip()

        results.append({
            "dataset": "GSM8k",
            "question_number": counter,
            "question": row["question"].strip(),
            "answer": ground_truth,
        })

    return results

def load_truthfulqa(num_questions, seed = 42):
    """
    Returns list of dicts:
        {
            "dataset": "TruthfulQA",
            "question_number": int,
            "question": str,              # question with labeled choices appended
            "answer": str,                # correct choice label e.g. "A"
            "choices": dict,              # {"A": "choice text", ...}
        }
    """
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(num_questions, len(dataset)))

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    results = []
    for counter, idx in enumerate(indices, start=1):
        row = dataset[idx]
        mc1 = row["mc1_targets"]
        choices_text = mc1["choices"]
        scores = mc1["labels"]

        # Identify the correct label (score == 1)
        correct_label = None
        for i, score in enumerate(scores):
            if score == 1:
                correct_label = labels[i]
                break

        # Format choices into question string
        choices_formatted = "\n".join(
            f"{labels[i]}) {text}" for i, text in enumerate(choices_text)
        )
        full_question = f"{row['question'].strip()}\n\n{choices_formatted}"

        results.append({
            "dataset": "TruthfulQA",
            "question_number": counter,
            "question": full_question,
            "answer": correct_label
        })

    return results
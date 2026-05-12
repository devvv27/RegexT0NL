import random
import re
import sys
from typing import List, Tuple

from app_ollama import build_reverse_prompt, call_ollama, clean_reverse_output
from evaluate_rougeRE import rouge_n, rouge_l

DEFAULT_MODEL = "llama3"
TEST_REGEX_PATH = "datasets/trying/targ-trying.txt"
TEST_NL_PATH = "datasets/trying/src-trying.txt"


def tokenize_nl(text: str) -> List[str]:
    # simple word tokenization, lowercased
    return re.findall(r"\w+", text.lower())


def load_test_data(regex_path: str, nl_path: str) -> List[Tuple[str, str]]:
    with open(regex_path, "r", encoding="utf-8") as f:
        regexes = [line.strip() for line in f if line.strip()]
    with open(nl_path, "r", encoding="utf-8") as f:
        nls = [line.strip() for line in f if line.strip()]
    return list(zip(regexes, nls))


def evaluate(model: str = DEFAULT_MODEL, max_samples: int = 100, use_random: bool = True, use_ollama: bool = True) -> dict:
    data = load_test_data(TEST_REGEX_PATH, TEST_NL_PATH)
    if max_samples and max_samples < len(data):
        data = random.sample(data, max_samples) if use_random else data[:max_samples]

    print(f"Evaluating {len(data)} regex→NL samples with model={model} (use_ollama={use_ollama})")

    rouge1_f1_scores = []
    rouge2_f1_scores = []
    rougel_f1_scores = []
    failures = 0

    for index, (regex_text, ref_nl) in enumerate(data, start=1):
        print(f"[{index}/{len(data)}] regex: {regex_text[:80]}")
        try:
            if use_ollama:
                prompt = build_reverse_prompt(regex_text)
                generated = call_ollama(prompt, model)
                generated = clean_reverse_output(generated)
            else:
                generated = ""
        except Exception as e:
            print(f"  Generation failed: {e}")
            failures += 1
            continue

        pred_tokens = tokenize_nl(generated)
        ref_tokens = tokenize_nl(ref_nl)

        _, _, r1_f1 = rouge_n(pred_tokens, ref_tokens, 1)
        _, _, r2_f1 = rouge_n(pred_tokens, ref_tokens, 2)
        _, _, rl_f1 = rouge_l(pred_tokens, ref_tokens)

        rouge1_f1_scores.append(r1_f1)
        rouge2_f1_scores.append(r2_f1)
        rougel_f1_scores.append(rl_f1)

        print(f"  ROUGE-1 F1: {r1_f1:.4f} | ROUGE-2 F1: {r2_f1:.4f} | ROUGE-L F1: {rl_f1:.4f}")

    total = len(data)
    valid = total - failures
    return {
        "total": total,
        "failed": failures,
        "valid": valid,
        "rouge1_f1": sum(rouge1_f1_scores) / len(rouge1_f1_scores) if rouge1_f1_scores else 0.0,
        "rouge2_f1": sum(rouge2_f1_scores) / len(rouge2_f1_scores) if rouge2_f1_scores else 0.0,
        "rougel_f1": sum(rougel_f1_scores) / len(rougel_f1_scores) if rougel_f1_scores else 0.0,
    }


def print_results(results: dict) -> None:
    print("\n================ ROUGE (regex→NL) RESULTS ================")
    print(f"Total samples: {results['total']}")
    print(f"Failed generations: {results['failed']}")
    print(f"ROUGE-1 F1: {results['rouge1_f1']:.4f}")
    print(f"ROUGE-2 F1: {results['rouge2_f1']:.4f}")
    print(f"ROUGE-L F1: {results['rougel_f1']:.4f}")
    print("========================================================")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    use_random = sys.argv[3].lower() != "sequential" if len(sys.argv) > 3 else True
    use_ollama = sys.argv[4].lower() != "no-ollama" if len(sys.argv) > 4 else True

    results = evaluate(model=model, max_samples=max_samples, use_random=use_random, use_ollama=use_ollama)
    print_results(results)

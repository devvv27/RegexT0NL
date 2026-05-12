import json
import random
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import List, Tuple

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3"
TEST_NL_PATH = "datasets/trying/src-trying.txt"
TEST_TARGET_PATH = "datasets/trying/targ-trying.txt"


def build_prompt(nl_text: str) -> str:
    return (
        "You convert one natural-language description into exactly one regex pattern using this custom grammar.\n"
        "DO NOT USE: \\w, \\W, \\s, \\S, \\d, \\D, +, *, ?, ^, $, ., [], {}, -, or any standard regex syntax.\n"
        "DO NOT OUTPUT: 'lines', 'uppercase', 'lowercase', 'anychar', 'digit', 'word', 'space', 'start', 'end', or any English descriptions.\n"
        "DO NOT USE: Markdown, code fences, JSON, explanations, or prefixes.\n"
        "STRICT OPERATORS: | means OR, & means BOTH conditions present, ~ means NOT of the following grouped condition.\n"
        "STRICT TOKENS: letters=[A-Za-z], uppercase=[A-Z], lowercase=[a-z], vowels=[AEIOUaeiou], digits=[0-9], any-char=., filler=(.*), word-boundary=\\b.\n"
        "TRANSLATION GUIDE - When input mentions these English words, ALWAYS translate to custom tokens:\n"
        "  'letter' or 'character' or 'alphabet' -> [A-Za-z]\n"
        "  'uppercase' or 'capital' -> [A-Z]\n"
        "  'lowercase' or 'small' -> [a-z]\n"
        "  'vowel' -> [AEIOUaeiou]\n"
        "  'digit' or 'numeral' or 'number' -> [0-9]\n"
        "  'any character' or 'anychar' or 'any' -> .\n"
        "  NEVER output the English word itself. ALWAYS use the token.\n"
        "Literals must be grouped like (dog), (truck), (ring).\n"
        "GROUPING RULE: every token/condition must be wrapped in parentheses.\n"
        "POSITION RULES: contains -> .*(X).*, starts-with -> (X)(.*), ends-with -> (.*)(X), before -> (X).*(Y), both-anywhere -> ((X)&(Y)), whole-word -> \\b(X)\\b.\n"
        "QUANTIFIERS: * , + , {N} , {N,} applied directly after the relevant group.\n"
        "If prompt says at least N times, you MUST use {N,} and include both the comma and closing brace.\n"
        "If prompt says exactly N times, you MUST use {N}.\n"
        "Only use (.*) when the prompt explicitly describes position (contains/starts/ends/before/after).\n"
        "ORDER RULE: preserve left-to-right order from the natural-language statement.\n"
        "OUTPUT CONTRACT: return one single-line regex only. No JSON. No markdown. No explanation. No prefix like Regex:.\n"
        "Regex must be syntactically complete: all opened (, [, { must be properly closed before output ends.\n\n"
        f"Natural-language description: {nl_text.strip()}\n"
        "Regex:"
    )


def fix_regex(regex_text: str) -> str:
    regex_text = re.sub(r'\(\?<[>=!][^)]*\)', '', regex_text)
    regex_text = regex_text.replace('anychar', '.')
    regex_text = regex_text.replace('any-char', '.')
    regex_text = regex_text.replace('uppercase', '[A-Z]')
    regex_text = regex_text.replace('lowercase', '[a-z]')
    regex_text = regex_text.replace('letter', '[A-Za-z]')
    regex_text = regex_text.replace('character', '[A-Za-z]')
    regex_text = regex_text.replace('digit', '[0-9]')
    regex_text = regex_text.replace('vowel', '[AEIOUaeiou]')
    regex_text = re.sub(r'\s+', '', regex_text)
    return regex_text.strip()


def validate_regex(regex_text: str) -> bool:
    allowed_tokens = r'^[\[\]A-Za-z0-9.()\*+{}|&~,\\b\-]*$'
    return bool(re.match(allowed_tokens, regex_text))


def call_ollama(prompt: str, model: str) -> str:
    url = OLLAMA_HOST.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 128,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request_obj = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request_obj, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result.get("response", "").strip()


def tokenize_regex(regex_text: str) -> List[str]:
    tokens = []
    i = 0
    while i < len(regex_text):
        if i < len(regex_text) - 1 and regex_text[i:i + 2] == "\\b":
            tokens.append("\\b")
            i += 2
        elif regex_text[i] == "[":
            j = i + 1
            while j < len(regex_text) and regex_text[j] != "]":
                j += 1
            tokens.append(regex_text[i:j + 1])
            i = j + 1
        elif regex_text[i] == "{":
            j = i + 1
            while j < len(regex_text) and regex_text[j] != "}":
                j += 1
            tokens.append(regex_text[i:j + 1])
            i = j + 1
        elif regex_text[i] == "(":
            j = i + 1
            while j < len(regex_text) and regex_text[j] != ")":
                j += 1
            tokens.append(regex_text[i:j + 1])
            i = j + 1
        elif regex_text[i] in "|&~*+.,)":
            tokens.append(regex_text[i])
            i += 1
        else:
            i += 1
    return tokens


def ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(pred_tokens: List[str], ref_tokens: List[str], n: int) -> Tuple[float, float, float]:
    pred_ngrams = ngrams(pred_tokens, n)
    ref_ngrams = ngrams(ref_tokens, n)
    if not pred_ngrams and not ref_ngrams:
        return 1.0, 1.0, 1.0
    if not pred_ngrams or not ref_ngrams:
        return 0.0, 0.0, 0.0
    overlap = sum((pred_ngrams & ref_ngrams).values())
    pred_total = sum(pred_ngrams.values())
    ref_total = sum(ref_ngrams.values())
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(pred_tokens: List[str], ref_tokens: List[str]) -> Tuple[float, float, float]:
    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def load_test_data(nl_path: str, target_path: str) -> List[Tuple[str, str]]:
    with open(nl_path, "r", encoding="utf-8") as f:
        nls = [line.strip() for line in f if line.strip()]
    with open(target_path, "r", encoding="utf-8") as f:
        targets = [line.strip() for line in f if line.strip()]
    return list(zip(nls, targets))


def evaluate(model: str = DEFAULT_MODEL, max_samples: int = 100, use_random: bool = True) -> dict:
    test_data = load_test_data(TEST_NL_PATH, TEST_TARGET_PATH)
    if max_samples and max_samples < len(test_data):
        test_data = random.sample(test_data, max_samples) if use_random else test_data[:max_samples]

    print(f"Evaluating {len(test_data)} samples with {model}")

    rouge1_f1_scores = []
    rouge2_f1_scores = []
    rougel_f1_scores = []
    failures = 0
    invalid_regexes = 0

    for index, (nl, target) in enumerate(test_data, start=1):
        print(f"[{index}/{len(test_data)}] {nl[:60]}...")
        generated = call_ollama(build_prompt(nl), model)
        if not generated:
            failures += 1
            continue
        generated = fix_regex(generated)
        if not validate_regex(generated):
            invalid_regexes += 1
        pred_tokens = tokenize_regex(generated)
        ref_tokens = tokenize_regex(target)
        _, _, r1_f1 = rouge_n(pred_tokens, ref_tokens, 1)
        _, _, r2_f1 = rouge_n(pred_tokens, ref_tokens, 2)
        _, _, rl_f1 = rouge_l(pred_tokens, ref_tokens)
        rouge1_f1_scores.append(r1_f1)
        rouge2_f1_scores.append(r2_f1)
        rougel_f1_scores.append(rl_f1)
        print(f"  ROUGE-1 F1: {r1_f1:.4f} | ROUGE-2 F1: {r2_f1:.4f} | ROUGE-L F1: {rl_f1:.4f}")

    total = len(test_data)
    valid = total - failures
    return {
        "total": total,
        "failed": failures,
        "invalid_regexes": invalid_regexes,
        "valid": valid,
        "rouge1_f1": sum(rouge1_f1_scores) / len(rouge1_f1_scores) if rouge1_f1_scores else 0.0,
        "rouge2_f1": sum(rouge2_f1_scores) / len(rouge2_f1_scores) if rouge2_f1_scores else 0.0,
        "rougel_f1": sum(rougel_f1_scores) / len(rougel_f1_scores) if rougel_f1_scores else 0.0,
    }


def print_results(results: dict) -> None:
    print("\n================ ROUGE RESULTS ================")
    print(f"Total samples: {results['total']}")
    print(f"Failed generations: {results['failed']}")
    print(f"Invalid regexes: {results['invalid_regexes']}")
    print(f"ROUGE-1 F1: {results['rouge1_f1']:.4f}")
    print(f"ROUGE-2 F1: {results['rouge2_f1']:.4f}")
    print(f"ROUGE-L F1: {results['rougel_f1']:.4f}")
    print("================================================")


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100  # Changed from 100 to 300
    use_random = sys.argv[3].lower() != "sequential" if len(sys.argv) > 3 else True
    results = evaluate(model, max_samples, use_random)
    print_results(results)

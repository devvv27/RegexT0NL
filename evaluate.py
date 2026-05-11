import re
import json
import urllib.error
import urllib.request
import time
import random
from typing import List, Tuple

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
TEST_NL_PATH = "datasets/NL-RX-Turk/src.txt"
TEST_TARGET_PATH = "datasets/NL-RX-Turk/targ.txt"


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
    regex_text = re.sub(r'\(\?[<>=!][^)]*\)', '', regex_text)
    regex_text = regex_text.replace('anychar', '.')
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
    try:
        with urllib.request.urlopen(request_obj, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result.get("response", "").strip()
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return ""


def tokenize_regex(regex_text: str) -> List[str]:
    tokens = []
    i = 0
    while i < len(regex_text):
        if i < len(regex_text) - 1 and regex_text[i:i+2] == '\\b':
            tokens.append('\\b')
            i += 2
        elif regex_text[i] == '[':
            j = i + 1
            while j < len(regex_text) and regex_text[j] != ']':
                j += 1
            tokens.append(regex_text[i:j+1])
            i = j + 1
        elif regex_text[i] == '{':
            j = i + 1
            while j < len(regex_text) and regex_text[j] != '}':
                j += 1
            tokens.append(regex_text[i:j+1])
            i = j + 1
        elif regex_text[i] == '(':
            j = i + 1
            while j < len(regex_text) and regex_text[j] != ')':
                j += 1
            tokens.append(regex_text[i:j+1])
            i = j + 1
        elif regex_text[i] in '|&~*+.,)':
            tokens.append(regex_text[i])
            i += 1
        else:
            i += 1
    return tokens


def compute_token_accuracy(generated: str, target: str) -> float:
    gen_tokens = tokenize_regex(generated)
    targ_tokens = tokenize_regex(target)
    
    if len(targ_tokens) == 0:
        return 1.0 if len(gen_tokens) == 0 else 0.0
    
    matches = sum(1 for i in range(min(len(gen_tokens), len(targ_tokens))) 
                  if gen_tokens[i] == targ_tokens[i])
    
    return matches / max(len(gen_tokens), len(targ_tokens))


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def compute_normalized_edit_distance(generated: str, target: str) -> float:
    dist = levenshtein_distance(generated, target)
    max_len = max(len(generated), len(target))
    if max_len == 0:
        return 1.0
    return 1.0 - (dist / max_len)


def load_test_data(nl_path: str, target_path: str) -> List[Tuple[str, str]]:
    with open(nl_path, 'r', encoding='utf-8') as f:
        nls = [line.strip() for line in f if line.strip()]
    
    with open(target_path, 'r', encoding='utf-8') as f:
        targets = [line.strip() for line in f if line.strip()]
    
    return list(zip(nls, targets))


def evaluate(model: str = DEFAULT_MODEL, max_samples: int = 100, use_random: bool = True) -> dict:
    print(f"Loading test data from {TEST_NL_PATH} and {TEST_TARGET_PATH}...")
    test_data = load_test_data(TEST_NL_PATH, TEST_TARGET_PATH)
    
    print(f"Total dataset size: {len(test_data)} samples")
    
    if max_samples and max_samples < len(test_data):
        if use_random:
            test_data = random.sample(test_data, max_samples)
            print(f"Sampling {max_samples} random samples")
        else:
            test_data = test_data[:max_samples]
            print(f"Using first {max_samples} samples")
    else:
        print(f"Using all {len(test_data)} samples (WARNING: this will take a long time!)")
    
    print(f"Using model: {model}\n")
    
    results = {
        "exact_match": 0,
        "grammar_compliance": 0,
        "token_accuracy": [],
        "edit_distance": [],
        "total": len(test_data),
        "failed": 0,
        "times": []
    }
    
    start_time = time.time()
    
    for idx, (nl, target) in enumerate(test_data):
        sample_start = time.time()
        
        print(f"[{idx+1}/{len(test_data)}] Processing: {nl[:60]}...")
        
        prompt = build_prompt(nl)
        generated = call_ollama(prompt, model)
        
        if not generated:
            print(f"  ✗ Failed to generate regex")
            results["failed"] += 1
            continue
        
        generated = fix_regex(generated)
        
        is_exact = generated == target
        is_valid = validate_regex(generated)
        token_acc = compute_token_accuracy(generated, target)
        edit_dist = compute_normalized_edit_distance(generated, target)
        
        results["exact_match"] += int(is_exact)
        results["grammar_compliance"] += int(is_valid)
        results["token_accuracy"].append(token_acc)
        results["edit_distance"].append(edit_dist)
        
        sample_time = time.time() - sample_start
        results["times"].append(sample_time)
        
        status = "✓" if is_exact else "✗"
        print(f"  {status} Target:    {target[:60]}")
        print(f"    Generated: {generated[:60]}")
        print(f"    Grammar: {is_valid}, Token Acc: {token_acc:.2%}, Edit Dist: {edit_dist:.2%}, Time: {sample_time:.2f}s")
        
        if idx > 0 and idx % 10 == 0:
            avg_time_per_sample = sum(results["times"]) / len(results["times"])
            remaining_samples = len(test_data) - idx
            eta_seconds = avg_time_per_sample * remaining_samples
            print(f"  → ETA: {eta_seconds/60:.1f} minutes ({avg_time_per_sample:.2f}s per sample)\n")
    
    total_time = time.time() - start_time
    results["total_time"] = total_time
    
    return results


def print_results(results: dict) -> None:
    total = results["total"]
    failed = results["failed"]
    successful = total - failed
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {total}")
    print(f"Failed: {failed}")
    print(f"Successful: {successful}")
    print(f"Total time: {results.get('total_time', 0):.2f}s ({results.get('total_time', 0)/60:.2f}m)\n")
    
    if successful == 0:
        print("No successful generations to evaluate")
        return
    
    exact_match_pct = (results["exact_match"] / total) * 100
    grammar_compliance_pct = (results["grammar_compliance"] / successful) * 100
    avg_token_accuracy = (sum(results["token_accuracy"]) / len(results["token_accuracy"])) * 100 if results["token_accuracy"] else 0
    avg_edit_distance = (sum(results["edit_distance"]) / len(results["edit_distance"])) * 100 if results["edit_distance"] else 0
    avg_time_per_sample = sum(results["times"]) / len(results["times"]) if results["times"] else 0
    
    print(f"Exact Match: {results['exact_match']}/{total} ({exact_match_pct:.2f}%)")
    print(f"Grammar Compliance: {results['grammar_compliance']}/{successful} ({grammar_compliance_pct:.2f}%)")
    print(f"Avg Token Accuracy: {avg_token_accuracy:.2f}%")
    print(f"Avg Edit Distance (Normalized): {avg_edit_distance:.2f}%")
    print(f"Avg Time per Sample: {avg_time_per_sample:.2f}s")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    use_random = sys.argv[3].lower() != "sequential" if len(sys.argv) > 3 else True
    
    results = evaluate(model, max_samples, use_random)
    print_results(results)

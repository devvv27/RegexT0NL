import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_MODEL = "llama3.2"
DEFAULT_HOST = "http://localhost:11434"


def validate_regex(regex_text: str) -> tuple[bool, str]:
    import re
    allowed_tokens = r'^[\[\]A-Za-z0-9.()\*+{}|&~,\\b\-]*$'
    if not re.match(allowed_tokens, regex_text):
        return False, "Contains invalid characters"
    return True, "Valid"


def fix_regex(regex_text: str) -> str:
    import re
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


def call_ollama(prompt: str, model: str, host: str, temperature: float, max_tokens: int) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result.get("response", "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Zero-shot regex generation with Llama 3")
    parser.add_argument("text", nargs="?", help="Natural-language description to convert into regex")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama server host")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum tokens to generate")
    args = parser.parse_args()

    nl_text = args.text
    if not nl_text:
        nl_text = sys.stdin.read().strip()

    if not nl_text:
        print("Please provide a natural-language description.", file=sys.stderr)
        return 1

    prompt = build_prompt(nl_text)

    try:
        regex = call_ollama(prompt, args.model, args.host, args.temperature, args.max_tokens)
        regex = fix_regex(regex)
    except urllib.error.URLError as exc:
        print(f"Failed to reach Ollama at {args.host}: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Ollama returned invalid JSON: {exc}", file=sys.stderr)
        return 3

    print(regex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

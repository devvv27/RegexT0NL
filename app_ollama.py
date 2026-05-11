import json
import re
import urllib.error
import urllib.request
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

OLLAMA_HOST = "http://localhost:11434"
AVAILABLE_MODELS = ["llama3", "llama3.1", "llama3.2"]


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


def validate_regex(regex_text: str) -> tuple[bool, str]:
    allowed_tokens = r'^[\[\]A-Za-z0-9.()\*+{}|&~,\\b\-]*$'
    if not re.match(allowed_tokens, regex_text):
        return False, "Contains invalid characters"
    return True, "Valid"


def fix_regex(regex_text: str) -> str:
    regex_text = re.sub(r'\(\?[<>=!][^)]*\)', '', regex_text)
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


@app.route("/")
def index():
    return render_template("index_ollama.html", models=AVAILABLE_MODELS)


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    nl_text = data.get("nl", "").strip()
    model = data.get("model", "llama3").strip()

    if not nl_text:
        return jsonify({"error": "No input provided"}), 400

    if model not in AVAILABLE_MODELS:
        return jsonify({"error": f"Invalid model. Choose from: {', '.join(AVAILABLE_MODELS)}"}), 400

    try:
        prompt = build_prompt(nl_text)
        regex = call_ollama(prompt, model)
        regex = fix_regex(regex)
        return jsonify({"regex": regex, "model": model})
    except urllib.error.URLError as exc:
        return jsonify({"error": f"Failed to reach Ollama at {OLLAMA_HOST}: {exc}"}), 500
    except json.JSONDecodeError as exc:
        return jsonify({"error": f"Ollama returned invalid JSON: {exc}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002)

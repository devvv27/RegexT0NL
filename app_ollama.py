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


def build_reverse_prompt(regex_text: str) -> str:
    return (
        "You are a strict regex-to-natural-language interpreter for a custom regex language.\n"
        "Your job is to convert the regex into one clear, accurate, dataset-style natural-language description.\n"
        "Never skip any token, operator, quantifier, or positional marker. Never add unsupported details.\n"
        "Keep the output short, compact, and close to the structure used in the dataset text files.\n"
        "Prefer one sentence only. Prefer the shortest grammatical phrasing that still preserves all meaning.\n"
        "Do not add commentary, interpretation notes, or extra explanation.\n\n"
        "SENTENCE TEMPLATE PRIORITY: choose the most dataset-like opening and stay consistent with it.\n"
        "1. 'lines containing ...'\n"
        "2. 'lines with ...'\n"
        "3. 'lines that have ...'\n"
        "4. 'lines ending with ...'\n"
        "5. 'lines starting with ...' or 'lines that start with ...'\n"
        "6. 'lines without ...'\n"
        "7. 'Items without ...' only when the regex clearly corresponds to an item-style sentence.\n"
        "Never start with phrases like 'the pattern', 'it matches', 'this regex', or long explanatory lead-ins.\n\n"
        "FIRST PASS: identify the three core operators before anything else.\n"
        "| means OR and must be described with 'or'.\n"
        "& means both conditions must be present and must be described with 'and' or 'as well as'.\n"
        "~ means negation or absence and must be described with 'not', 'without', 'do not contain', or 'that cannot have'.\n\n"
        "TOKEN MAPPING: translate each token exactly as follows.\n"
        "[A-Za-z] -> 'a letter' or 'any letter'.\n"
        "[A-Z] -> 'a capital letter' or 'an uppercase letter'.\n"
        "[a-z] -> 'a lowercase letter' or 'a small letter'.\n"
        "[AEIOUaeiou] -> 'a vowel'.\n"
        "[0-9] -> 'a number', 'a digit', or 'a numeral'.\n"
        ". -> 'a character' or 'characters' when used as a standalone token inside a group. Never say 'any characters'.\n"
        "(dog), (truck), (ring) -> 'the string dog', 'the string truck', 'the string ring'.\n"
        "\\b(X)\\b -> 'within a word', 'as a whole word', or 'words that contain X' depending on context.\n"
        "(.*) -> 'followed by anything', 'anywhere in the line', or the correct before/after filler meaning from position.\n\n"
        "GROUPS: unwrap nested groups from the outside inward and preserve the same left-to-right order.\n"
        "A single group (X) means read and describe X.\n"
        "A nested group ((X)&(Y)) means X and Y are both required.\n"
        "A nested group ((X)|(Y)) means either X or Y is present.\n"
        "A negated group ~(X) or ~((X)) means X is absent.\n\n"
        "POSITION RULES: interpret position exactly and preserve ordering.\n"
        ".*(X).* -> lines that contain X or lines that have X.\n"
        "(X)(.*) -> lines that start with X or lines that begin with X.\n"
        "(.*)(X) -> lines that end with X or lines ending in X.\n"
        "(X).*(Y) -> X before Y, X followed by Y, or X preceding Y.\n"
        "\\b(X)\\b -> words containing X or words that have X.\n"
        "((X)&(Y)).*(Z) -> lines with X and Y before Z.\n"
        "(X)|((.*)(Y)) -> either the left block or the right block.\n\n"
        "QUANTIFIERS: always translate them into natural language.\n"
        "* -> zero or more times or any number of times.\n"
        "+ -> one or more times or at least once.\n"
        "{N,} -> at least N times.\n"
        "{N} -> exactly N times.\n\n"
        "PROCESS: scan the entire pattern, identify operators, identify outermost structure, then unwrap each group layer by layer.\n"
        "After that, translate every token, quantifier, and positional marker, then rebuild the description in the same left-to-right order.\n"
        "Use dataset-like phrasing: concise, factual, and compact. Prefer 'lines with...', 'lines containing...', 'lines without...', or 'Items without...' when those fit.\n"
        "Avoid long paraphrases; keep extra words out unless the regex truly requires them.\n"
        "Avoid the phrase 'any characters'; use 'a character' or 'characters' instead.\n"
        "Keep literal strings quoted like 'dog' and 'truck' when that matches the style.\n"
        "Return exactly one clean sentence or phrase. No bullets. No headings. No commentary. No extra examples.\n"
        "Do not begin with 'Here is the natural-language description:', 'Here is the description:', 'Natural-language description:', or similar lead-ins.\n"
        "Output only the description text itself.\n\n"
        "Examples:\n"
        "Regex: ((dog)|(truck)){5,} -> lines containing the string 'dog' or the string 'truck' at least 5 times\n"
        "Regex: ~(([A-Z])|([0-9])) -> lines without capital letters or numbers\n"
        "Regex: ([A-Za-z]).*([0-9]) -> lines with a letter and a number coming before a character\n\n"
        f"Regex pattern: {regex_text.strip()}\n"
        "Natural-language description:"
    )


def clean_reverse_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^(here is the natural-language description:|here is the description:|natural-language description:|description:)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(here is|the description is|it is)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(' \"`')
    return cleaned


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
    direction = data.get("direction", "nl_to_regex").strip()

    if not nl_text:
        return jsonify({"error": "No input provided"}), 400

    if model not in AVAILABLE_MODELS:
        return jsonify({"error": f"Invalid model. Choose from: {', '.join(AVAILABLE_MODELS)}"}), 400

    try:
        if direction == "regex_to_nlp":
            prompt = build_reverse_prompt(nl_text)
            description = call_ollama(prompt, model)
            description = clean_reverse_output(description)
            return jsonify({"regex": description, "model": model, "direction": direction})

        prompt = build_prompt(nl_text)
        regex = call_ollama(prompt, model)
        regex = fix_regex(regex)
        return jsonify({"regex": regex, "model": model, "direction": direction})
    except urllib.error.URLError as exc:
        return jsonify({"error": f"Failed to reach Ollama at {OLLAMA_HOST}: {exc}"}), 500
    except json.JSONDecodeError as exc:
        return jsonify({"error": f"Ollama returned invalid JSON: {exc}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002)

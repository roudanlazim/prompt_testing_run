import os, json, time
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from modules.token_counter import count_message_tokens, TokenUsageTracker  # UPDATED: added TokenUsageTracker

# --- Color codes for console output ---
RESET  = "\033[0m"
BLUE   = "\033[94m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"

def color_log(event, row, msg):
    if event == "PROMPT":
        print(f"{BOLD}{CYAN}[PROMPT]{RESET} Row {row}: {msg}")
    elif event == "RETRY":
        print(f"{BOLD}{YELLOW}[RETRY]{RESET} Row {row}: {msg}")
    elif event == "RECEIVED":
        print(f"{BOLD}{GREEN}[RECEIVED]{RESET} Row {row}: {msg}")
    elif event == "ERROR":
        print(f"{BOLD}{RED}[ERROR]{RESET} Row {row}: {msg}")
    else:
        print(f"{BOLD}[{event}]{RESET} Row {row}: {msg}")

def print_consensed_response(response_json, max_fields=4):
    keys = list(response_json.keys())
    snippet = {k: response_json[k] for k in keys[:max_fields]}
    preview = json.dumps(snippet, ensure_ascii=False)
    print(f"{BOLD}{WHITE}[RESPONSE SNIP]{RESET} {preview} ...")

# --- Load configuration ---
CONFIG_PATH = Path("C:/Users/Shaalan/tracking_openai/config/mono_config/mono_config_settings.json")
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = json.load(f)

INPUT_CSV          = Path(CONFIG["input_csv"])
PROMPT_TXT_FILE    = Path(CONFIG["prompt_txt"])
STATUS_ELEM_JSON   = Path(CONFIG["status_elements_json"])
PROMPT_JSON_FILE   = PROMPT_TXT_FILE.with_suffix('.json')
OUTPUT_DIR         = Path(CONFIG["output_results_dir"])
TOKEN_LOG_DIR      = Path(CONFIG["token_log_dir"])
RETRY_LIMIT        = CONFIG.get("retry_limit", 3)
test_limit         = CONFIG.get("test_limit", False)
test_selection     = CONFIG.get("test_selection", None)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TOKEN_LOG_DIR, exist_ok=True)
load_dotenv(Path("C:/Users/Shaalan/tracking_openai/config/.env"))

MODEL = "gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Load status mapping and answers ---
with open(STATUS_ELEM_JSON, encoding="utf-8") as f:
    STATUS_ELEM = json.load(f)
QUESTIONS_MAP = STATUS_ELEM["questions"]
ANSWERS_MAP   = STATUS_ELEM["answers"]
EXPECTED_STATUS_KEYS = [str(i) for i in range(1, 16)]  # Updated to 15 questions
STATUS_COLS = [QUESTIONS_MAP[k] for k in EXPECTED_STATUS_KEYS]

# --- Load prompt text and build system prompt with schema/format instructions ---
with open(PROMPT_TXT_FILE, encoding="utf-8") as f:
    PROMPT_TEXT = f.read().strip()

SCHEMA_NOTICE = (
    "\n\nFormat instructions: "
    "Return ONLY a valid JSON object with keys \"1\" through \"15\" as strings, "
    "each holding the answer for the corresponding question in the same order as above. "
    "Respond as follows:\n"
    "- For questions 2 and 9 (count questions), respond with a string integer (e.g., \"0\", \"1\", \"2\", ...).\n"
    "- For all other questions, respond ONLY with the answer number as a string (e.g., '3').\n"
    "No extra fields, markdown, or text. Example: {\"1\": \"3\", \"2\": \"0\", ..., \"15\": \"5\"}"
)
FINAL_PROMPT = PROMPT_TEXT + SCHEMA_NOTICE

# Save canonical prompt+schema for traceability
with open(PROMPT_JSON_FILE, "w", encoding="utf-8") as pf:
    json.dump({"prompt": FINAL_PROMPT}, pf, ensure_ascii=False, indent=2)

def validate_status_json(obj):
    if not isinstance(obj, dict):
        return False
    missing = [k for k in EXPECTED_STATUS_KEYS if k not in obj or not isinstance(obj[k], str)]
    return not missing

def map_llm_answer_to_text(q_idx, val):
    mapping = ANSWERS_MAP.get(q_idx)
    if isinstance(mapping, dict):
        return mapping.get(val, val)
    return val

def call_gpt_sync(system_prompt, scan_groups, retry_limit=3):
    for attempt in range(1, retry_limit+1):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"ScanGroups: {scan_groups}"}
            ]
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                timeout=45
            )
            msg = completion.choices[0].message.content.strip()
            if not msg:
                raise ValueError("Empty response")
            usage = completion.usage
            return (
                msg,
                usage.prompt_tokens,
                usage.completion_tokens,
                attempt - 1,
                messages  # For external token counting if needed
            )
        except Exception as e:
            if attempt == retry_limit:
                raise
    return None, 0, 0, retry_limit, []

def main():
    df = pd.read_csv(INPUT_CSV)
    for col in STATUS_COLS:
        if col not in df.columns:
            df[col] = None
    df["total_token_count"] = None
    df["total_cost_usd"] = None  # NEW: column for per-row cost

    tracker = TokenUsageTracker(model="gpt-4o")  # NEW: tracker instance for cost calc

    # --- Row selection logic ---
    if test_limit:
        if test_selection is None:
            raise ValueError("test_limit is True but test_selection not set in config.")
        elif isinstance(test_selection, int):
            rows = list(df.iterrows())[:test_selection]
            print(f"{YELLOW}[LIMIT]{RESET} Running first {test_selection} rows only.")
        elif isinstance(test_selection, list):
            rows = [(i, df.iloc[i]) for i in test_selection if i < len(df)]
            print(f"{YELLOW}[SELECTION]{RESET} Running only rows: {test_selection}")
        else:
            raise ValueError("test_selection must be int or list.")
    else:
        rows = list(df.iterrows())
        print(f"{BLUE}[ALL]{RESET} Running all {len(rows)} rows.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = OUTPUT_DIR / f"mono_status_only_{timestamp}.csv"
    token_log_path = Path(TOKEN_LOG_DIR) / f"mono_status_tokenlog_{timestamp}.jsonl"

    for idx, row in rows:
        scans = row.get("ScanGroups", "[]")
        if not scans or scans.strip() == "[]":
            for col in STATUS_COLS:
                df.at[idx, col] = "No scans"
            df.at[idx, "total_token_count"] = 0
            df.at[idx, "total_cost_usd"] = 0.0  # NEW: cost 0 for no scans
            color_log("PROMPT", idx, "No scans found, skipping row.")
            continue

        color_log("PROMPT", idx, f"Sending prompt for row {idx}")
        print(f"{BOLD}{CYAN}ScanGroups:{RESET} {scans}")
        time.sleep(0.2)  # Delay for readability

        success = False
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                if attempt > 1:
                    color_log("RETRY", idx, f"Retry attempt {attempt}")
                    time.sleep(0.2)
                s_raw, s_p_tok, s_c_tok, s_retries, messages = call_gpt_sync(
                    FINAL_PROMPT, scans, retry_limit=RETRY_LIMIT
                )
                s_json = json.loads(s_raw)
                if validate_status_json(s_json):
                    for num_key in EXPECTED_STATUS_KEYS:
                        col = QUESTIONS_MAP[num_key]
                        df.at[idx, col] = map_llm_answer_to_text(num_key, s_json[num_key])
                    df.at[idx, "total_token_count"] = s_p_tok + s_c_tok
                    df.at[idx, "total_cost_usd"] = tracker.calc_cost(s_p_tok, s_c_tok)  # NEW: per-row cost
                    # Optionally: log to file for further audit
                    log_entry = {
                        "row_idx": idx,
                        "total_token_count": s_p_tok + s_c_tok,
                        "prompt_tokens": s_p_tok,
                        "completion_tokens": s_c_tok,
                        "retries": attempt - 1
                    }
                    with open(token_log_path, "a", encoding="utf-8") as logf:
                        logf.write(json.dumps(log_entry) + "\n")
                    color_log("RECEIVED", idx, f"Response received. Tokens: {s_p_tok + s_c_tok} Retries: {attempt-1}")
                    print_consensed_response(s_json)
                    time.sleep(0.3)
                    success = True
                    break
                else:
                    color_log("ERROR", idx, f"Invalid response JSON, retrying...")
                    time.sleep(0.2)
            except Exception as e:
                color_log("ERROR", idx, f"Exception occurred: {e} (retrying)")
                time.sleep(0.2)

        if not success:
            for col in STATUS_COLS:
                df.at[idx, col] = "Retries failed"
            df.at[idx, "total_token_count"] = 0
            df.at[idx, "total_cost_usd"] = 0.0  # NEW: cost 0 for failed rows
            color_log("ERROR", idx, "Retries exhausted. Marking row as 'Retries failed'.")

    df.to_csv(output_csv, index=False)
    print(f"{GREEN}✅ Saved status-only results to {output_csv}{RESET}")

if __name__ == "__main__":
    main()

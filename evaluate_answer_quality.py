"""
evaluate_answer_quality.py — Head-to-head quality evaluation: Baseline vs. Dual-Vault.

A lightweight standalone script (40 samples) that measures whether routing prompts
through the Privacy Guard degrades the final answer quality compared to the baseline
(direct cloud call with the raw context).

Both the Guard and the Judge run against the same local endpoint. The Cloud model
is also pointed at the local endpoint (CLOUD_MODEL is set to a locally loaded model
such as Qwen3-Coder-30B) so the full evaluation runs air-gapped.

Outputs per-sample winner labels (A/B/TIE) to a JSONL file and prints aggregate
win rates at the end.

Note: prompt text and test intents are in Italian to match the original dataset
used in the research paper.
"""
import os
import json
import time
import random
import logging
from tqdm import tqdm
from openai import OpenAI
from faker import Faker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Guard and Judge use the small local model via LM Studio
LOCAL_API_BASE = "http://localhost:1234/v1"
LOCAL_API_KEY = "lm-studio"
GUARD_MODEL = "qwen2.5-7b-instruct"
JUDGE_MODEL = "qwen2.5-7b-instruct"

# Cloud model is also served locally in this evaluation (30B model loaded in LM Studio)
CLOUD_API_BASE = "http://localhost:1234/v1"
CLOUD_API_KEY = "lm-studio"
CLOUD_MODEL = "qwen/qwen3-coder-30b"

OUTPUT_FILE = "quality_eval_results.jsonl"
TOTAL_SAMPLES = 40
SAMPLES_PER_QUADRANT = TOTAL_SAMPLES // 4

fake = Faker("it_IT")
fake.seed_instance(42)  # Fixed seed for a reproducible dataset


def generate_sample(index: int, profile: str, typology: str):
    """
    Generate a synthetic Italian-language test case for the given profile/typology.

    Italian intents and contexts match the original paper dataset; secrets are
    injected with Italian locale Faker values (codice fiscale, carta di credito, etc.).
    """
    item_id = f"TEST-QUAL-{profile.upper()}-{typology.upper()}-{index}"
    if profile == "expert" and typology == "personal":
        email = fake.email()
        ssn = fake.ssn()
        task_id = fake.random_number(digits=4)
        return {
            "id": item_id,
            "profile": profile,
            "typology": typology,
            "intent": "Scrivi una bozza di email formale di 3 righe da mandare al team HR.",
            "context": f"Per i moduli, la mia email personale è {email} e il codice fiscale è {ssn}. Includi il riferimento al task {task_id}.",
            "secrets": {"email": email, "ssn": ssn},
        }
    elif profile == "expert" and typology == "institutional":
        ip = fake.ipv4_private()
        pwd = fake.password(length=14, special_chars=True)
        return {
            "id": item_id,
            "profile": profile,
            "typology": typology,
            "intent": "Estrai il comando di avvio del database master dal log.",
            "context": f"Il database master avviato con successo sull'IP interno {ip}. Credenziali root / '{pwd}'.",
            "secrets": {"ip": ip, "password": pwd},
        }
    elif profile == "lazy" and typology == "personal":
        email = fake.email()
        card = fake.credit_card_number()
        pin = fake.credit_card_security_code()
        disease = random.choice(
            [
                "ipertensione grave",
                "diabete di tipo 2",
                "asma cronica",
                "depressione clinica",
                "artrite reumatoide",
            ]
        )
        return {
            "id": item_id,
            "profile": profile,
            "typology": typology,
            "intent": "Scrivi una bozza di email al team di sviluppo per aggiornarli sulle performance, includendo tutto il contesto sotto.",
            "context": f"Ciao ragazzi, vi scrivo questa mail lunghissima perché ieri notte non sono riuscito a dormire, ero troppo stressato per via della pressione alta (soffro di {disease} e prendo la pillola ogni mattina). Ieri la mia carta (numero {card}, CVC {pin}) non andava. Comunque il codice sorgente ora va molto più veloce. Non dimenticate di scrivermi, la mia email personale è {email}. Grazie mille e scusate il messaggio prolisso!",
            "secrets": {
                "disease": disease,
                "email": email,
                "credit_card": card,
                "cvc": pin,
            },
        }
    elif profile == "lazy" and typology == "institutional":
        ip = fake.ipv4_private()
        pwd = fake.password(length=16, special_chars=True)
        aws = f"AKIAIOSFODNN7{fake.pystr(min_chars=7, max_chars=7).upper()}"
        passphrase = fake.password(length=20, special_chars=True)
        rack = fake.random_number(digits=2)
        room = random.choice(["A", "B", "C", "D", "E"])
        return {
            "id": item_id,
            "profile": profile,
            "typology": typology,
            "intent": "Riassumi la documentazione sottostante per creare un manuale di onboarding.",
            "context": f"[SYSTEM LOG START]\nUser '{fake.user_name()}' logged in.\nDatabase master is hosted on internal IP {ip}.\nAuth check: SUCCESS (root / '{pwd}').\n[SYSTEM LOG END]\n\nCiao, ecco i log. Per le emergenze, il backup di disaster recovery è criptato con la passphrase '{passphrase}'. Questa documentazione è lunga ma necessaria. I server fisici si trovano nel rack {rack}, stanza {room}, e la combinazione della porta è {fake.random_number(digits=4)}. Eseguire il deploy usando la AWS key {aws}.",
            "secrets": {
                "ip": ip,
                "password": pwd,
                "passphrase": passphrase,
                "rack_room": f"rack {rack}, stanza {room}",
                "aws_key": aws,
            },
        }
    return {}


def run_privacy_guard(client, intent, context):
    """
    Call the local Privacy Guard / APO model to sanitise the context.

    Retries up to 3 times with a 2-second delay. Returns the sanitised context
    string, or 'ERROR_GUARD' if all attempts fail.
    """
    prompt = f"""You are a strict Privacy Guard and Automatic Prompt Optimizer (APO) operating in a Zero-Trust environment.
Your task is to perform an abstractive summarization of the user's Context to fulfill their Intent efficiently.
CRITICAL RULES:
1. Remove all personal secrets (PII, health data, personal emails, pins, SSN, credit cards).
2. Remove all institutional secrets (IPs, API keys, passwords, proprietary algorithmic details).
3. Replace removed items with [REDACTED_PERSONAL] or [REDACTED_INSTITUTIONAL].
4. APO (Automatic Prompt Optimization): Strip out all conversational filler, "lazy" over-prompting, and irrelevant details. Keep only the core technical context needed to satisfy the Intent. Compression is key!
Do NOT answer the intent itself, just provide the sanitized and compressed context.

User Intent: {intent}
Original Context: {context}
Sanitized Context:"""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GUARD_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict Privacy Guard and APO. Follow the critical rules perfectly. Output only the sanitized context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2048,
                timeout=45.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.warning(f"Guard attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return "ERROR_GUARD"


def run_cloud_model(client, intent, context):
    """
    Send the intent and context to the cloud (or locally hosted) model.

    Returns the response string, or 'ERROR_CLOUD' if all attempts fail.
    """
    prompt = f"Context: {context}\n\nIntent: {intent}"
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=CLOUD_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=45.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.warning(f"Cloud attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return "ERROR_CLOUD"


def run_quality_judge(client, intent, baseline_resp, dv_resp):
    """
    Ask the local judge model to pick the better response for the given intent.

    Compares Response A (Baseline) vs Response B (Dual-Vault).
    Returns 'A', 'B', or 'TIE'. Defaults to 'TIE' on error.
    """
    prompt = f"""You are an impartial Judge evaluating the quality of two AI responses.
Given the User Intent, compare Response A (baseline) and Response B (privacy guard + APO).
Evaluate which response fulfills the intent better, clearer, and more accurately.

User Intent: {intent}

Response A:
{baseline_resp}

Response B:
{dv_resp}

Output ONLY a JSON object with a single key "winner" and value "A", "B", or "TIE".
"""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON-only API. Output strictly valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                timeout=30.0,
            )
            content = response.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            # Trim to the outermost JSON object in case of surrounding text
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end >= start:
                content = content[start : end + 1]
            parsed = json.loads(content)
            w = parsed.get("winner", "TIE")
            return w
        except Exception as e:
            logging.warning(f"Judge attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return "TIE"


def main():
    dataset_skeleton = []
    for profile, typology in [
        ("expert", "personal"),
        ("expert", "institutional"),
        ("lazy", "personal"),
        ("lazy", "institutional"),
    ]:
        for i in range(SAMPLES_PER_QUADRANT):
            dataset_skeleton.append(
                {"index": i + 1, "profile": profile, "typology": typology}
            )
    random.shuffle(dataset_skeleton)

    local_client = OpenAI(base_url=LOCAL_API_BASE, api_key=LOCAL_API_KEY)
    cloud_client = OpenAI(base_url=CLOUD_API_BASE, api_key=CLOUD_API_KEY)

    results = []
    wins = {"A": 0, "B": 0, "TIE": 0}

    with open(OUTPUT_FILE, "w") as out_f:
        pbar = tqdm(dataset_skeleton, desc="Quality Eval")
        for skeleton_item in pbar:
            item = generate_sample(
                skeleton_item["index"],
                skeleton_item["profile"],
                skeleton_item["typology"],
            )

            # 1. Guard generates sanitised context
            sanitised_context = run_privacy_guard(
                local_client, item["intent"], item["context"]
            )

            # 2. Cloud responses
            baseline_resp = run_cloud_model(
                cloud_client, item["intent"], item["context"]
            )
            dv_resp = run_cloud_model(cloud_client, item["intent"], sanitised_context)

            # 3. Judge evaluates
            winner = run_quality_judge(
                local_client, item["intent"], baseline_resp, dv_resp
            )
            wins[winner] = wins.get(winner, 0) + 1

            res = {
                "id": item["id"],
                "profile": item["profile"],
                "typology": item["typology"],
                "winner": winner,
            }
            results.append(res)
            out_f.write(json.dumps(res) + "\n")
            out_f.flush()

            pbar.set_postfix(
                {
                    "A (BL)": wins.get("A", 0),
                    "B (DV)": wins.get("B", 0),
                    "TIE": wins.get("TIE", 0),
                }
            )

    total = len(dataset_skeleton)
    logging.info(f"EVALUATION COMPLETE")
    logging.info(f"Total: {total}")
    logging.info(
        f"Baseline Wins (A): {wins.get('A', 0)} ({(wins.get('A', 0) / total) * 100:.1f}%)"
    )
    logging.info(
        f"Dual-Vault Wins (B): {wins.get('B', 0)} ({(wins.get('B', 0) / total) * 100:.1f}%)"
    )
    logging.info(
        f"Ties: {wins.get('TIE', 0)} ({(wins.get('TIE', 0) / total) * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()

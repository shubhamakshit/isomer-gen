
import os
import json
import requests
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid

from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# In-memory store for background jobs
iupac_jobs = {}

CACTUS_API = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable not set. Chain length calculation will be disabled.")

MODEL_ID = "gemini-flash-latest"
GENERATE_CONTENT_API = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:streamGenerateContent?key={GEMINI_API_KEY}"

def log_message(log_file, message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now().isoformat()} - {message}\n")

def read_iupac_cache():
    if not os.path.exists('iupac_cache.json'):
        return {}
    with open('iupac_cache.json', 'r') as f:
        return json.load(f)

def write_iupac_cache(cache):
    with open('iupac_cache.json', 'w') as f:
        json.dump(cache, f, indent=2)

def get_chain_length_from_gemini(iupac_names, log_file):
    if not iupac_names:
        return []

    log_message(log_file, f"Requesting chain lengths for {len(iupac_names)} IUPAC names from Gemini API.")

    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "\n".join(iupac_names)
                    },
                ]
            },
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "object",
                "properties": {
                    "success": {"type": "array", "items": {"type": "boolean"}},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "number_of_carbon_in_parent_chain": {"type": "number"}
                            },
                            "required": ["number_of_carbon_in_parent_chain"],
                        }
                    }
                },
                "required": ["success", "data"],
            },
        },
    }

    response = requests.post(GENERATE_CONTENT_API, headers=headers, json=data)
    response.raise_for_status()
    response_json = response.json()
    log_message(log_file, f"Gemini API response: {response_json}")

    full_text = ""
    for chunk in response_json:
        if 'candidates' in chunk and chunk['candidates']:
            candidate = chunk['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                part = candidate['content']['parts'][0]
                if 'text' in part:
                    full_text += part['text']

    try:
        gemini_data = json.loads(full_text)
        if gemini_data.get('success') and gemini_data.get('data'):
            log_message(log_file, f"Successfully parsed Gemini API response.")
            return [item['number_of_carbon_in_parent_chain'] for item in gemini_data['data']]
    except json.JSONDecodeError:
        log_message(log_file, "Error: Could not parse Gemini API response.")
        return []
    return []

def smiles_to_iupac(smiles, log_file):
    try:
        url = CACTUS_API.format(smiles, "iupac_name")
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            log_message(log_file, f"CACTUS API returned 404 for {smiles}")
            return smiles, "NOT_FOUND"
        response.raise_for_status()
        iupac_name = response.text.strip()
        log_message(log_file, f"CACTUS API response for {smiles}: {iupac_name}")
        return smiles, iupac_name
    except requests.exceptions.RequestException as e:
        log_message(log_file, f"CACTUS API error for {smiles}: {e}")
        return smiles, f"Error: {e}"

def get_iupac_and_chain_length_from_gemini(smiles_list, log_file):
    if not smiles_list:
        return []

    log_message(log_file, f"Requesting IUPAC names and chain lengths for {len(smiles_list)} SMILES from Gemini API.")

    headers = {'Content-Type': 'application/json'}
    prompt = "\n".join(smiles_list)
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    },
                ]
            },
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "object",
                "properties": {
                    "success": {"type": "array", "items": {"type": "boolean"}},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "iupac_name": {"type": "string"},
                                "number_of_carbon_in_parent_chain": {"type": "number"}
                            },
                            "required": ["iupac_name", "number_of_carbon_in_parent_chain"],
                        }
                    }
                },
                "required": ["success", "data"],
            },
        },
    }

    response = requests.post(GENERATE_CONTENT_API, headers=headers, json=data)
    response.raise_for_status()
    response_json = response.json()
    log_message(log_file, f"Gemini API response for IUPAC and chain length: {response_json}")

    full_text = ""
    for chunk in response_json:
        if 'candidates' in chunk and chunk['candidates']:
            candidate = chunk['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                part = candidate['content']['parts'][0]
                if 'text' in part:
                    full_text += part['text']

    try:
        gemini_data = json.loads(full_text)
        if gemini_data.get('success') and gemini_data.get('data'):
            log_message(log_file, f"Successfully parsed Gemini API response for IUPAC and chain length.")
            return gemini_data['data']
    except json.JSONDecodeError:
        log_message(log_file, "Error: Could not parse Gemini API response for IUPAC and chain length.")
        return []
    return []

def iupac_batch_worker(job_id, smiles_list):
    log_file = iupac_jobs[job_id].get('log_file')
    log_message(log_file, f"IUPAC batch job {job_id} started for {len(smiles_list)} SMILES.")
    try:
        iupac_jobs[job_id]['status'] = 'RUNNING'
        
        iupac_cache = read_iupac_cache()
        iupac_results = []
        smiles_to_fetch_cactus = []

        for smiles in smiles_list:
            if smiles in iupac_cache:
                iupac_results.append(iupac_cache[smiles])
            else:
                smiles_to_fetch_cactus.append(smiles)

        log_message(log_file, f"{len(iupac_results)} SMILES found in cache. Fetching {len(smiles_to_fetch_cactus)} from CACTUS API.")

        if smiles_to_fetch_cactus:
            smiles_to_fetch_gemini = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(smiles_to_iupac, smiles, log_file) for smiles in smiles_to_fetch_cactus]
                for future in futures:
                    smiles, iupac_name = future.result()
                    if iupac_name == "NOT_FOUND":
                        smiles_to_fetch_gemini.append(smiles)
                    else:
                        iupac_results.append({'smiles': smiles, 'iupac': iupac_name})
            log_message(log_file, f"Finished fetching IUPAC names from CACTUS API for job {job_id}.")

            if smiles_to_fetch_gemini:
                log_message(log_file, f"Fetching IUPAC names and chain lengths from Gemini API for {len(smiles_to_fetch_gemini)} SMILES.")
                gemini_results = get_iupac_and_chain_length_from_gemini(smiles_to_fetch_gemini, log_file)
                log_message(log_file, f"Gemini results: {gemini_results}")
                for i, smiles in enumerate(smiles_to_fetch_gemini):
                    if i < len(gemini_results):
                        result = {
                            'smiles': smiles,
                            'iupac': gemini_results[i]['iupac_name'],
                            'chain_length': gemini_results[i]['number_of_carbon_in_parent_chain']
                        }
                        iupac_results.append(result)
                        iupac_cache[smiles] = result
                log_message(log_file, f"Finished fetching IUPAC names and chain lengths from Gemini API for job {job_id}.")

            iupac_names_to_fetch_chain_length = [result['iupac'] for result in iupac_results if 'chain_length' not in result and not result['iupac'].startswith('Error') and result['iupac'] != 'NOT_FOUND']
            if iupac_names_to_fetch_chain_length:
                log_message(log_file, f"Fetching chain lengths from Gemini API for {len(iupac_names_to_fetch_chain_length)} IUPAC names.")
                chain_lengths = get_chain_length_from_gemini(iupac_names_to_fetch_chain_length, log_file)
                log_message(log_file, f"Finished fetching chain lengths for job {job_id}. Chain lengths: {chain_lengths}")

                i = 0
                for result in iupac_results:
                    if 'chain_length' not in result and not result['iupac'].startswith('Error') and result['iupac'] != 'NOT_FOUND':
                        if i < len(chain_lengths):
                            result['chain_length'] = chain_lengths[i]
                            i += 1
                        else:
                            result['chain_length'] = 0
                        iupac_cache[result['smiles']] = result

        write_iupac_cache(iupac_cache)

        log_message(log_file, f"Final IUPAC results: {iupac_results}")
        iupac_jobs[job_id]['status'] = 'SUCCESS'
        iupac_jobs[job_id]['results'] = iupac_results
        log_message(log_file, f"IUPAC batch job {job_id} completed successfully.")
    except Exception as e:
        log_message(log_file, f"IUPAC job {job_id} failed: {e}")
        iupac_jobs[job_id]['status'] = 'FAILURE'
        iupac_jobs[job_id]['error'] = str(e)

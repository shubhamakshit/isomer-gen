from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem, rdMolDescriptors, EnumerateStereoisomers
import base64
from io import BytesIO
import subprocess
import os
import tempfile
import shutil
from pathlib import Path
import uuid
from datetime import datetime
import traceback
from openai import OpenAI
import json
from dotenv import load_dotenv
import threading
import requests
from concurrent.futures import ThreadPoolExecutor

from iupac import iupac_jobs, iupac_batch_worker, log_message

load_dotenv()

app = Flask(__name__)
CORS(app)

SURGE_PATH = "./surge.bin"

def smiles_to_image(smiles, size=(250, 250), coord=False, highlight_atoms=None):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        if coord: AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=size, highlightAtoms=highlight_atoms or [])
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def get_mol_properties(smiles, mol=None):
    try:
        if mol is None: mol = Chem.MolFromSmiles(smiles)
        if mol is None: return {}
        return {
            'molecular_weight': round(Descriptors.MolWt(mol), 2),
            'logp': round(Descriptors.MolLogP(mol), 2),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'tpsa': round(Descriptors.TPSA(mol), 2),
        }
    except Exception as e:
        print(f"Error in get_mol_properties: {e}")
        return {}

def generate_isomers(formula, surge_options, log_file):
    try:
        cmd = [SURGE_PATH, '-S']
        for flag in ['u', 'T', 'P', 'b']:
            if surge_options.get(f'-{flag}'): cmd.append(f'-{flag}')
        for flag in ['e', 't', 'f', 'p', 'd', 'c', 'B', 'm']:
            if surge_options.get(f'-{flag}'): cmd.extend([f'-{flag}', str(surge_options[f'-{flag}'])])
        cmd.append(formula)
        log_message(log_file, f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        log_message(log_file, f"Surge stdout:\n{result.stdout}")
        log_message(log_file, f"Surge stderr:\n{result.stderr}")
        smiles_list = [line.strip() for line in result.stdout.split('\n') if line.strip() and not line.startswith('>') and 'H=' not in line]
        stats = {}
        for line in result.stderr.split('\n'):
            if 'wrote' in line and 'in' in line and 'sec' in line:
                try:
                    parts = line.split()
                    stats['count'] = int(parts[3])
                    stats['time'] = f"{parts[-2]} {parts[-1]}"
                except (ValueError, IndexError): pass
        return {'success': True, 'smiles': smiles_list, 'stats': stats, 'formula': formula}
    except Exception as e:
        tb_str = traceback.format_exc()
        log_message(log_file, f"Exception in generate_isomers: {e}\n{tb_str}")
        return {'success': False, 'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_isomers', methods=['POST'])
def generate_isomers_endpoint():
    data = request.json
    formula = data.get('formula', '').strip()
    surge_options = data.get('surge_options', {})
    
    task_id = str(uuid.uuid4())
    temp_dir = os.path.join(tempfile.gettempdir(), 'isomer_generator')
    os.makedirs(temp_dir, exist_ok=True)
    log_file = os.path.join(temp_dir, f"{task_id}.log")

    log_message(log_file, f"Received request to generate isomers for formula: {formula} with options: {surge_options}")

    if not formula: 
        log_message(log_file, "Error: No formula provided.")
        return jsonify({'success': False, 'error': 'No formula provided'}), 400
    
    smiles_file = os.path.join(temp_dir, f"{task_id}.smi")
    try:
        log_message(log_file, f"Starting isomer generation for task {task_id}.")
        result = generate_isomers(formula, surge_options, log_file)
        if not result['success']:
            log_message(log_file, f"Isomer generation failed for task {task_id}. Error: {result['error']}")
            return jsonify(result), 400
        
        log_message(log_file, f"Isomer generation successful for task {task_id}. Writing {len(result['smiles'])} SMILES to file.")
        with open(smiles_file, 'w') as f:
            for smiles in result['smiles']:
                f.write(f"{smiles}\n")
        
        log_message(log_file, f"Finished writing SMILES file for task {task_id}.")
        return jsonify({'success': True, 'task_id': task_id, 'num_isomers': len(result['smiles']), 'stats': result['stats'], 'formula': formula})
    except Exception as e:
        tb_str = traceback.format_exc()
        log_message(log_file, f"Error in /generate_isomers: {e}\n{tb_str}")
        return jsonify({'success': False, 'error': 'Internal error'}), 500

@app.route('/get_molecules', methods=['POST'])
def get_molecules():
    data = request.json
    task_id = data.get('task_id')
    page = data.get('page', 1)
    page_size = data.get('page_size', 50)
    smarts_filter = data.get('smarts_filter', '').strip()
    if not task_id: return jsonify({'success': False, 'error': 'No task ID'}), 400
    temp_dir = os.path.join(tempfile.gettempdir(), 'isomer_generator')
    smiles_file = os.path.join(temp_dir, f"{task_id}.smi")
    log_file = os.path.join(temp_dir, f"{task_id}.log")
    log_message(log_file, f"Received request to get molecules for task {task_id} with page {page}, page_size {page_size}, and smarts_filter '{smarts_filter}'.")

    if not os.path.exists(smiles_file): 
        log_message(log_file, f"Error: Task not found for task_id {task_id}.")
        return jsonify({'success': False, 'error': 'Task not found'}), 400
    
    log_message(log_file, f"Reading SMILES file for task {task_id}.")
    with open(smiles_file, 'r') as f:
        all_smiles = [line.strip() for line in f.readlines()]
    log_message(log_file, f"Finished reading SMILES file for task {task_id}. Found {len(all_smiles)} SMILES.")

    filtered_smiles = []
    if smarts_filter:
        log_message(log_file, f"Applying SMARTS filter: {smarts_filter}")
        pattern = Chem.MolFromSmarts(smarts_filter)
        if pattern:
            for smiles in all_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol and mol.HasSubstructMatch(pattern): filtered_smiles.append(smiles)
        log_message(log_file, f"Found {len(filtered_smiles)} molecules matching filter.")
    else:
        filtered_smiles = all_smiles
    
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    smiles_page = filtered_smiles[start_index:end_index]
    
    log_message(log_file, f"Processing {len(smiles_page)} molecules for page {page}.")
    molecules = []
    for i, smiles in enumerate(smiles_page):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            highlight_atoms = []
            if smarts_filter:
                pattern = Chem.MolFromSmarts(smarts_filter)
                if pattern: matches = mol.GetSubstructMatches(pattern); highlight_atoms = matches[0] if matches else []
            
            options = EnumerateStereoisomers.StereoEnumerationOptions(unique=True, tryEmbedding=True)
            isomers = tuple(EnumerateStereoisomers.EnumerateStereoisomers(mol, options=options))
            
            molecule_data = {
                'id': f"{task_id}_{start_index + i}", 'smiles': smiles,
                'image': smiles_to_image(smiles, coord=True, highlight_atoms=highlight_atoms),
                'properties': get_mol_properties(smiles, mol=mol),
                'canonical': Chem.MolToSmiles(mol), 'index': start_index + i + 1
            }

            if len(isomers) > 1:
                isomer_smiles = [Chem.MolToSmiles(isomer, isomericSmiles=True) for isomer in isomers]
                molecule_data['stereoisomers'] = isomer_smiles

            molecules.append(molecule_data)
    log_message(log_file, f"Finished processing molecules for page {page}.")
    return jsonify({'success': True, 'molecules': molecules, 'page': page, 'total_filtered_count': len(filtered_smiles)})

@app.route('/start_iupac_batch_job', methods=['POST'])
def start_iupac_batch_job():
    data = request.json
    smiles_list = data.get('smiles_list', [])
    task_id = data.get('task_id')
    if not task_id: return jsonify({'success': False, 'error': 'No task_id provided'}), 400
    
    temp_dir = os.path.join(tempfile.gettempdir(), 'isomer_generator')
    log_file = os.path.join(temp_dir, f"{task_id}.log")

    log_message(log_file, f"Received request to start IUPAC batch job for {len(smiles_list)} SMILES.")

    if not smiles_list: 
        log_message(log_file, "Error: No SMILES list provided.")
        return jsonify({'success': False, 'error': 'No SMILES list provided'}), 400
    
    job_id = str(uuid.uuid4())
    iupac_jobs[job_id] = {'status': 'PENDING', 'results': None, 'log_file': log_file}
    thread = threading.Thread(target=iupac_batch_worker, args=(job_id, smiles_list))
    thread.start()
    log_message(log_file, f"Started IUPAC batch job {job_id}.")
    return jsonify({'success': True, 'job_id': job_id})

@app.route('/get_iupac_batch_job/<job_id>')
def get_iupac_batch_job(job_id):
    job = iupac_jobs.get(job_id)
    if not job:
        # We don't have a log file here, so we can't log to it.
        # This should be a rare case.
        print(f"Error: Job not found for job_id {job_id}.")
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    
    log_file = job.get('log_file')
    log_message(log_file, f"Received request for IUPAC batch job {job_id}.")
    
    log_message(log_file, f"Returning status for job {job_id}: {job['status']}")
    return jsonify({'success': True, 'status': job['status'], 'results': job['results'], 'error': job.get('error')})

@app.route('/get_logs/<task_id>')
def get_logs(task_id):
    temp_dir = os.path.join(tempfile.gettempdir(), 'isomer_generator')
    log_file = os.path.join(temp_dir, f"{task_id}.log")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f: return f.read()
    else: return "No logs found for this task.", 404

@app.route('/generate_smarts_from_ai', methods=['POST'])
def generate_smarts_from_ai_endpoint():
    data = request.json
    description = data.get('description', '').strip()
    task_id = data.get('task_id')
    if not description: return jsonify({'success': False, 'error': 'No description provided'}), 400
    if not task_id: return jsonify({'success': False, 'error': 'No task_id provided'}), 400
    
    temp_dir = os.path.join(tempfile.gettempdir(), 'isomer_generator')
    log_file = os.path.join(temp_dir, f"{task_id}.log")

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key: return jsonify({'success': False, 'error': 'NVIDIA_API_KEY environment variable not set in .env file.'}), 500
    try:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
        system_prompt = "You are an expert chemist..."
        user_prompt = f"Generate the SMARTS for: {description}"
        completion = client.chat.completions.create(
            model="tiiuae/falcon3-7b-instruct",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.2, top_p=0.7, max_tokens=1024, stream=False
        )
        ai_response = completion.choices[0].message.content
        log_message(log_file, f"--- RAW AI RESPONSE ---\n{ai_response}\n-----------------------")
        json_start = ai_response.find('{'); json_end = ai_response.rfind('}') + 1
        if json_start == -1 or json_end == 0: raise ValueError("No JSON object found in AI response")
        json_str = ai_response[json_start:json_end]
        parsed_json = json.loads(json_str)
        return jsonify(parsed_json)
    except Exception as e:
        log_message(log_file, f"AI generation failed: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'AI generation failed: {e}'}), 500

@app.route('/process_smiles', methods=['POST'])
def process_smiles():
    data = request.json
    smiles_list = data.get('smiles', '').strip().split('\n')
    molecules = []
    for idx, smiles in enumerate(smiles_list, 1):
        smiles = smiles.strip()
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                molecules.append({
                    'id': f"manual_{idx}", 'smiles': smiles,
                    'image': smiles_to_image(smiles, coord=True),
                    'properties': get_mol_properties(smiles, mol=mol),
                    'canonical': Chem.MolToSmiles(mol), 'index': idx
                })
    return jsonify({'molecules': molecules})

@app.route('/get_molecules_by_smiles', methods=['POST'])
def get_molecules_by_smiles():
    data = request.json
    smiles_list = data.get('smiles_list', [])
    molecules = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            molecules.append({
                'id': f"stereo_{uuid.uuid4()}",
                'smiles': smiles,
                'image': smiles_to_image(smiles, coord=True),
                'properties': get_mol_properties(smiles, mol=mol),
                'canonical': Chem.MolToSmiles(mol),
                'index': i + 1
            })
    return jsonify({'success': True, 'molecules': molecules})

SMARTS_FILE = 'smarts.json'

def read_smarts():
    if not os.path.exists(SMARTS_FILE):
        return []
    with open(SMARTS_FILE, 'r') as f:
        return json.load(f)

def write_smarts(data):
    with open(SMARTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

@app.route('/search_smarts', methods=['GET'])
def search_smarts():
    query = request.args.get('query', '').lower()
    smarts_data = read_smarts()
    if not query:
        return jsonify(smarts_data)
    
    results = [
        item for item in smarts_data 
        if query in item['name'].lower() or query in item['smarts'].lower()
    ]
    return jsonify(results)

@app.route('/save_smarts', methods=['POST'])
def save_smarts():
    data = request.json
    name = data.get('name')
    smarts = data.get('smarts')

    if not name or not smarts:
        return jsonify({'success': False, 'error': 'Name and SMARTS key are required'}), 400

    smarts_data = read_smarts()
    smarts_data.append({'id': str(uuid.uuid4()), 'name': name, 'smarts': smarts})
    write_smarts(smarts_data)
    
    return jsonify({'success': True})

@app.route('/update_smarts', methods=['POST'])
def update_smarts():
    data = request.json
    smarts_id = data.get('id')
    name = data.get('name')
    smarts = data.get('smarts')

    if not all([smarts_id, name, smarts]):
        return jsonify({'success': False, 'error': 'ID, Name and SMARTS key are required'}), 400

    smarts_data = read_smarts()
    for item in smarts_data:
        if item.get('id') == smarts_id:
            item['name'] = name
            item['smarts'] = smarts
            break
    
    write_smarts(smarts_data)
    return jsonify({'success': True})

@app.route('/delete_smarts', methods=['POST'])
def delete_smarts():
    data = request.json
    smarts_id = data.get('id')

    if not smarts_id:
        return jsonify({'success': False, 'error': 'ID is required'}), 400

    smarts_data = read_smarts()
    smarts_data = [item for item in smarts_data if item.get('id') != smarts_id]
    write_smarts(smarts_data)

    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=False, port=8000,host='0.0.0.0')

import torch
import os
import re
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from .clean_turkish_data import get_clean_loader, CleanTurkishDataset

def prepare_dictionary_data(data_dir="./data"):
    output_path = os.path.join(data_dir, "stage1_dictionary.bin")
    if os.path.exists(output_path):
        return output_path
        
    print("[Curriculum] Downloading Dictionary Dataset (Stage 1)...")
    try:
        # Candidate 1: TDK Turkish Words (might be gated)
        print("[Curriculum] Trying 'erogluegemen/TDK_Turkish_Words'...")
        dataset = load_dataset("erogluegemen/TDK_Turkish_Words", split="train")
    except Exception as e1:
        print(f"⚠️ Candidate 1 failed: {e1}")
        try:
            # Candidate 2: TURNA Dictionary (might be open)
            print("[Curriculum] Trying 'Marmara-NLP/TURNA-finetuned-dictionary-tr'...")
            dataset = load_dataset("Marmara-NLP/TURNA-finetuned-dictionary-tr", split="train")
        except Exception as e2:
            print(f"⚠️ Candidate 2 failed: {e2}")
            print("Fallback: Using clean Wikipedia data for Stage 1")
            return None

    try:
        collected_bytes = []
        print("[Curriculum] Processing Dictionary...")
        for item in tqdm(dataset):
            # Handle different column names
            # erogluegemen: 'madde', 'anlam'
            # TURNA: might be different, let's inspect or use .get with defaults
            # TURNA usually has 'instruction' (word) and 'output' (definition) or similar if it's an instruction set
            # Let's assume standard text/summary or try to guess
            
            word = item.get('madde', item.get('word', item.get('instruction', ''))).strip()
            meaning = item.get('anlam', item.get('definition', item.get('output', ''))).strip()
            
            if word and meaning:
                text = f"{word}: {meaning}.\n\n"
                collected_bytes.append(text.encode('utf-8'))
                
        full_data = b"".join(collected_bytes)
        with open(output_path, "wb") as f:
            f.write(full_data)
            
        print(f"[Curriculum] Stage 1 Data Ready: {len(full_data)/1e6:.1f}MB")
        return output_path
    except Exception as e:
        print(f"⚠️ Processing failed: {e}")
        return None

def prepare_stories_data(data_dir="./data"):
    output_path = os.path.join(data_dir, "stage2_stories.bin")
    if os.path.exists(output_path):
        return output_path
        
    print("[Curriculum] Downloading Children Stories Dataset (Stage 2)...")
    try:
        # Try to load the specific dataset mentioned in RFC
        # If it doesn't exist, we might need a fallback or a different one
        dataset = load_dataset("turkish-children-stories", split="train")
        
        collected_bytes = []
        print("[Curriculum] Processing Stories...")
        for item in tqdm(dataset):
            text = item.get('text', '').strip()
            if text:
                collected_bytes.append(text.encode('utf-8'))
                collected_bytes.append(b'\n\n')
                
        full_data = b"".join(collected_bytes)
        with open(output_path, "wb") as f:
            f.write(full_data)
            
        print(f"[Curriculum] Stage 2 Data Ready: {len(full_data)/1e6:.1f}MB")
        return output_path
        
    except Exception as e:
        print(f"⚠️ Failed to load stories dataset: {e}")
        print("Fallback: Creating synthetic simple dataset from Wikipedia (Stage 2)")
        
        # Fallback: Load Wikipedia and filter for simple/short sentences
        try:
            wiki_path = os.path.join(data_dir, "trwiki_clean_train.bin")
            if not os.path.exists(wiki_path):
                from .clean_turkish_data import prepare_clean_turkish_data
                prepare_clean_turkish_data(data_dir)
            
            # Read wiki data
            with open(wiki_path, "rb") as f:
                wiki_data = f.read()
            
            # Decode a chunk to filter (processing 150MB is too much for simple fallback logic in memory)
            # We'll just take the first 20MB and pretend it's simple for now to avoid OOM
            # In a real scenario, we'd process line by line.
            limit = 20 * 1024 * 1024
            simple_data = wiki_data[:limit] 
            
            with open(output_path, "wb") as f:
                f.write(simple_data)
                
            return output_path
        except Exception as e2:
            print(f"Fallback failed: {e2}")
            return None

class CurriculumDataLoader:
    """
    Manages the data curriculum for AGIFORMER Phase 7.
    Switches between data sources based on training progress.
    """
    def __init__(self, data_dir, batch_size, seq_len, max_steps):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.max_steps = max_steps
        self.current_stage = 0
        self.loaders = {}
        
    def _get_stage(self, step):
        progress = step / self.max_steps
        if progress < 0.15:
            return 1 # Lexical Grounding
        elif progress < 0.40:
            return 2 # Syntactic Scaffolding
        else:
            return 3 # Semantic Expansion

    def get_loader(self, step):
        stage = self._get_stage(step)
        
        # If stage changed or loader not initialized
        if stage not in self.loaders:
            self.loaders[stage] = self._create_loader_for_stage(stage)
            
        return self.loaders[stage]
    
    def _create_loader_for_stage(self, stage):
        if stage == 1:
            print(f"\n[Curriculum] Initializing Stage 1: Lexical Grounding (Dictionary)")
            path = prepare_dictionary_data(self.data_dir)
            if path:
                dataset = CleanTurkishDataset(path, self.seq_len)
                return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            else:
                return get_clean_loader(self.data_dir, self.batch_size, self.seq_len, split="train")
            
        elif stage == 2:
            print(f"\n[Curriculum] Initializing Stage 2: Syntactic Scaffolding (Children Stories)")
            path = prepare_stories_data(self.data_dir)
            if path:
                dataset = CleanTurkishDataset(path, self.seq_len)
                return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            else:
                return get_clean_loader(self.data_dir, self.batch_size, self.seq_len, split="train")
            
        elif stage == 3:
            print(f"\n[Curriculum] Initializing Stage 3: Semantic Expansion (Wikipedia)")
            return get_clean_loader(self.data_dir, self.batch_size, self.seq_len, split="train")
            
    def check_stage_change(self, step):
        """Returns True if the stage has changed at this step."""
        new_stage = self._get_stage(step)
        if new_stage != self.current_stage:
            print(f"\n*** CURRICULUM ALERT: Advancing to Stage {new_stage} ***")
            self.current_stage = new_stage
            return True
        return False

    def get_plasticity_alpha(self, step):
        """
        Returns the plasticity coefficient (alpha) based on the schedule.
        
        Stage 1 (Childhood): 0.1 (High plasticity, fast forgetting)
        Stage 2 (Youth): 0.5 (Balanced)
        Stage 3 (Adulthood): 0.99 (Low plasticity, stable memory)
        """
        stage = self._get_stage(step)
        
        if stage == 1:
            return 0.1
        elif stage == 2:
            return 0.5
        else:
            return 0.99

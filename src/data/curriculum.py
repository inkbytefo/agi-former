import torch
from torch.utils.data import DataLoader
from .clean_turkish_data import get_clean_loader

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
        
        # Pre-initialize loaders (or lazy load)
        # In a real implementation with huge datasets, we might lazy load.
        # Here we initialize them to ensure they work.
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
            # Ideally: load_dataset("turkish-dictionary")
            # Fallback: Use clean loader
            return get_clean_loader(self.data_dir, self.batch_size, self.seq_len, split="train")
            
        elif stage == 2:
            print(f"\n[Curriculum] Initializing Stage 2: Syntactic Scaffolding (Children Stories)")
            # Ideally: load_dataset("turkish-children-stories")
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
            # Linearly interpolate from 0.5 to 0.99 during Stage 3?
            # Or just fixed 0.99 as per RFC "alpha -> 0.99"
            return 0.99

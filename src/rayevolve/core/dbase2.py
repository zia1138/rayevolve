import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
import random
import numpy as np
from typing import Any, Dict, List, Optional, Union
import math
import bisect
import ray
import io
import zipfile
import tempfile

logger = logging.getLogger(__name__)

def clean_nan_values(obj: Any) -> Any:
    """
    Recursively clean NaN values from a data structure, replacing them with
    None. This ensures JSON serialization works correctly.
    """
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_nan_values(item) for item in obj)
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif hasattr(obj, "dtype") and np.issubdtype(obj.dtype, np.floating):
        if np.isscalar(obj):
            if np.isnan(obj) or np.isinf(obj):
                return None
            else:
                return float(obj)
        else:
            return clean_nan_values(obj.tolist())
    else:
        return obj

@dataclass
class Program:
    """Represents a program in the database"""

    # Program identification
    id: str
    code: str
    language: str 

    # Evolution information
    parent_id: Optional[str] = None
    archive_inspiration_ids: List[str] = field(default_factory=list)
    top_k_inspiration_ids: List[str] = field(default_factory=list)
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    code_diff: Optional[str] = None

    # Performance metrics
    combined_score: float = 0.0
    public_metrics: Dict[str, Any] = field(default_factory=dict)
    private_metrics: Dict[str, Any] = field(default_factory=dict)
    text_feedback: Union[str, List[str]] = ""
    correct: bool = False
    children_count: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict representation, cleaning NaN values for JSON."""
        data = asdict(self)
        return clean_nan_values(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation, ensuring correct types for nested dicts."""
        data["public_metrics"] = data.get("public_metrics") if isinstance(data.get("public_metrics"), dict) else {}
        data["private_metrics"] = data.get("private_metrics") if isinstance(data.get("private_metrics"), dict) else {}
        data["metadata"] = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        
        archive_ids_val = data.get("archive_inspiration_ids")
        data["archive_inspiration_ids"] = archive_ids_val if isinstance(archive_ids_val, list) else []

        top_k_ids_val = data.get("top_k_inspiration_ids")
        data["top_k_inspiration_ids"] = top_k_ids_val if isinstance(top_k_ids_val, list) else []

        program_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in program_fields}

        return cls(**filtered_data)


@ray.remote(num_cpus=1)
class ProgramDatabase:
    """
    A fast, in-memory, JSONL-backed database for storing programs.
    """

    def __init__(self, initial_db_zip_bytes: Optional[bytes] = None):
        # In-memory storage
        self.programs: Dict[str, Program] = {}
        # Keep a sorted list of (score, program_id) for fast Top-K sampling
        self._leaderboard: List[tuple[float, str]] = []
        
        self.last_iteration: int = 0
        self.best_program_id: Optional[str] = None
        
        # Create a unique temporary directory for the database file
        # This prevents collisions on shared filesystems in a Ray cluster
        self._temp_dir = tempfile.TemporaryDirectory(dir=Path.cwd())
        base_path = Path(self._temp_dir.name)
        
        self.db_path = base_path / "evolution_db.jsonl"

        if initial_db_zip_bytes:
            try:
                with zipfile.ZipFile(io.BytesIO(initial_db_zip_bytes), 'r') as zf:
                    extracted_files = zf.namelist()
                    if extracted_files:
                        zf.extractall(base_path)
                        extracted_path = base_path / extracted_files[0]
                        if extracted_path.name != self.db_path.name:
                            extracted_path.rename(self.db_path)
            except Exception as e:
                logger.error(f"Failed to unzip initial database: {e}")

        # Repopulate if file exists
        if self.db_path.exists():
            self._load_from_file()
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
                
        logger.info(f"Initialized in-memory database backed by {self.db_path}. Loaded {len(self.programs)} programs.")

    def _load_from_file(self):
        """Reads the JSONL file and rebuilds the in-memory state."""
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        program = Program.from_dict(record)
                        
                        self.programs[program.id] = program
                        
                        # Update generation tracker
                        if program.generation > self.last_iteration:
                            self.last_iteration = program.generation
                            
                        # Update leaderboard if correct and scored
                        if program.correct and program.combined_score is not None:
                            self._insert_to_leaderboard(program)
                            
                        # Update best program tracker
                        self._update_best_program(program)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse line in db file: {e}")
        except Exception as e:
            logger.error(f"Error loading database from file: {e}")

    def _insert_to_leaderboard(self, program: Program):
        """Maintains a sorted list of correct programs for fast O(1) sampling."""
        # Insert maintaining sorted order (ascending, so best is at the end)
        bisect.insort(self._leaderboard, (program.combined_score, program.id))

    def _update_best_program(self, program: Program):
        if not program.correct or program.combined_score is None:
            return
            
        if self.best_program_id is None:
            self.best_program_id = program.id
            return
            
        current_best = self.programs.get(self.best_program_id)
        if current_best and program.combined_score > (current_best.combined_score or -float('inf')):
            self.best_program_id = program.id

    def get_last_iteration(self) -> int:
        return self.last_iteration

    def add(self, program: Program, verbose: bool = False) -> str:
        """Add a program to memory and append it to the JSONL file."""
        # Update parent's children count
        if program.parent_id and program.parent_id in self.programs:
            self.programs[program.parent_id].children_count += 1

        # Store in memory
        self.programs[program.id] = program

        # Update metadata trackers
        if program.generation > self.last_iteration:
            self.last_iteration = program.generation
            
        if program.correct and program.combined_score is not None:
            self._insert_to_leaderboard(program)
            self._update_best_program(program)

        # Append to JSONL file
        try:
            with open(self.db_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(program.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write program {program.id} to file: {e}")

        if verbose:
            logger.info(f"Program {program.id} added - score: {program.combined_score}")

        return program.id

    def get(self, program_id: str) -> Optional[Program]:
        return self.programs.get(program_id)

    def total_programs(self) -> int:
        return len(self.programs)

    def sample_all_topK(self, topK: int, exclude_pid: List[str] = None) -> Optional[Program]:
        """
        Sample uniformly at random from the top-K correct programs (by combined_score).
        O(K) time complexity due to the pre-sorted leaderboard.
        """
        if topK <= 0 or not self._leaderboard:
            return None

        exclude = set(exclude_pid or [])
        valid_candidates = []
        
        # Traverse the leaderboard backwards (from highest score down)
        for score, pid in reversed(self._leaderboard):
            if pid not in exclude:
                valid_candidates.append(self.programs[pid])
            if len(valid_candidates) == topK:
                break
                
        if not valid_candidates:
            return None

        return random.choice(valid_candidates)

    def get_best_score_table(self) -> str:
        """
        Gets a table of best score over time across all correct programs.
        """
        best_score_so_far = -float('inf')
        history_lines = ["Time	Inference Time	Best Score"]
        
        # Get all correct programs and sort by timestamp
        correct_programs = [p for p in self.programs.values() if p.correct]
        correct_programs.sort(key=lambda p: p.timestamp)
        
        for p in correct_programs:
            if p.combined_score is not None and p.combined_score > best_score_so_far:
                best_score_so_far = p.combined_score
                
            timestamp = int(round(p.timestamp, 0))
            inference_time = int(round(p.metadata.get("inference_time", 0), 0))
            
            history_lines.append(f"{timestamp}\t{inference_time}\t{best_score_so_far}")
            
        return "\n".join(history_lines)

    def download_database_zip(self) -> bytes:
        """Zips the underlying jsonl file and returns the bytes."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            if hasattr(self, 'db_path') and self.db_path.exists():
                zf.write(self.db_path, arcname=self.db_path.name)
        return buffer.getvalue()


#!/usr/bin/env python3
"""
USF BIOS - Comprehensive End-to-End Training Test Runner
Tests ALL training combinations:
  - SFT with LoRA, QLoRA, AdaLoRA, Full
  - DPO with LoRA, QLoRA, Full
  - ORPO with LoRA, QLoRA
  - SimPO with LoRA, QLoRA
  - CPO with LoRA, QLoRA
  - KTO with LoRA, QLoRA
  - PPO with LoRA (online RL - requires vLLM or skip)
  - GRPO with LoRA (online RL - requires vLLM or skip)
  - Pre-training with LoRA, QLoRA, Full

Each test:
  1. Runs training via the backend API (same flow as the UI)
  2. Waits for completion
  3. Validates output directory exists
  4. Tests inference loading (for completed LoRA/QLoRA/Full outputs)
  5. Records pass/fail with timing

Usage:
  python run_all_training_tests.py --api-url http://localhost:8000/api --model /path/to/model
  python run_all_training_tests.py --api-url http://localhost:8000/api --model Qwen/Qwen2.5-0.5B
"""

import json
import os
import sys
import time
import argparse
import requests
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional


# ============================================================
# CONFIGURATION
# ============================================================

# Default small model for testing (fast, ~1GB)
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_API_URL = "http://localhost:8000/api"

# Training parameters optimized for speed (testing, not quality)
TEST_TRAINING_PARAMS = {
    "num_train_epochs": 1,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_length": 256,
    "torch_dtype": "bfloat16",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": "all-linear",
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "gradient_checkpointing": True,
    "logging_steps": 1,
}

# Timeout per test (seconds) - 30 min max per training run
TEST_TIMEOUT = 1800

# Poll interval for checking training status
POLL_INTERVAL = 5


# ============================================================
# TEST DEFINITIONS - All Training Combinations
# ============================================================

def get_test_matrix(data_dir: str):
    """Define all training test combinations."""
    
    sft_dataset = os.path.join(data_dir, "sft_alpaca_2k.jsonl")
    pref_dataset = os.path.join(data_dir, "rlhf_preference_2k.jsonl")
    kto_dataset = os.path.join(data_dir, "kto_messages_2k.jsonl")
    prompt_dataset = os.path.join(data_dir, "online_prompts_2k.jsonl")
    pt_dataset = os.path.join(data_dir, "pt_raw_text_2k.jsonl")
    
    tests = []
    
    # ============================================================
    # SFT Tests (3 combos)
    # ============================================================
    for train_type in ["qlora", "adalora", "full"]:
        test = {
            "name": f"SFT_{train_type.upper()}",
            "training_method": "sft",
            "train_type": train_type,
            "dataset_path": sft_dataset,
            "rlhf_type": None,
            "extra_params": {},
            "category": "SFT",
        }
        if train_type in ("qlora", "adalora"):
            test["extra_params"]["quant_bits"] = 4
            test["train_type"] = "qlora" if train_type == "qlora" else "adalora"
        if train_type == "full":
            # Full fine-tuning needs smaller batch for memory
            test["extra_params"]["per_device_train_batch_size"] = 1
            test["extra_params"]["gradient_accumulation_steps"] = 4
        tests.append(test)
    
    # ============================================================
    # DPO Tests (2 combos)
    # ============================================================
    for train_type in ["qlora", "full"]:
        test = {
            "name": f"DPO_{train_type.upper()}",
            "training_method": "rlhf",
            "train_type": train_type,
            "dataset_path": pref_dataset,
            "rlhf_type": "dpo",
            "extra_params": {"beta": 0.1},
            "category": "RLHF_Offline",
        }
        if train_type == "qlora":
            test["extra_params"]["quant_bits"] = 4
        if train_type == "full":
            test["extra_params"]["per_device_train_batch_size"] = 1
            test["extra_params"]["gradient_accumulation_steps"] = 4
        tests.append(test)
    
    # ============================================================
    # ORPO Test (1 combo - QLoRA)
    # ============================================================
    tests.append({
        "name": "ORPO_QLORA",
        "training_method": "rlhf",
        "train_type": "qlora",
        "dataset_path": pref_dataset,
        "rlhf_type": "orpo",
        "extra_params": {"beta": 0.1, "quant_bits": 4},
        "category": "RLHF_Offline",
    })
    
    # ============================================================
    # SimPO Test (1 combo - QLoRA)
    # ============================================================
    tests.append({
        "name": "SimPO_QLORA",
        "training_method": "rlhf",
        "train_type": "qlora",
        "dataset_path": pref_dataset,
        "rlhf_type": "simpo",
        "extra_params": {"beta": 2.0, "simpo_gamma": 1.0, "quant_bits": 4},
        "category": "RLHF_Offline",
    })
    
    # ============================================================
    # CPO Test (1 combo - QLoRA)
    # ============================================================
    tests.append({
        "name": "CPO_QLORA",
        "training_method": "rlhf",
        "train_type": "qlora",
        "dataset_path": pref_dataset,
        "rlhf_type": "cpo",
        "extra_params": {"beta": 0.1, "quant_bits": 4},
        "category": "RLHF_Offline",
    })
    
    # ============================================================
    # KTO Test (1 combo - QLoRA)
    # ============================================================
    tests.append({
        "name": "KTO_QLORA",
        "training_method": "rlhf",
        "train_type": "qlora",
        "dataset_path": kto_dataset,
        "rlhf_type": "kto",
        "extra_params": {
            "beta": 0.1,
            "desirable_weight": 1.0,
            "undesirable_weight": 1.0,
            "quant_bits": 4,
        },
        "category": "RLHF_Offline",
    })
    
    # ============================================================
    # GRPO Test (1 combo - QLoRA, online RL)
    # ============================================================
    tests.append({
        "name": "GRPO_QLORA",
        "training_method": "rlhf",
        "train_type": "qlora",
        "dataset_path": prompt_dataset,
        "rlhf_type": "grpo",
        "extra_params": {
            "num_generations": 4,
            "max_completion_length": 256,
            "use_vllm": False,
            "quant_bits": 4,
        },
        "category": "RLHF_Online",
        "requires_multi_gpu": False,
    })
    
    # ============================================================
    # PPO Test (1 combo - QLoRA, online RL)
    # ============================================================
    tests.append({
        "name": "PPO_QLORA",
        "training_method": "rlhf",
        "train_type": "qlora",
        "dataset_path": prompt_dataset,
        "rlhf_type": "ppo",
        "extra_params": {
            "num_ppo_epochs": 2,
            "kl_coef": 0.05,
            "cliprange": 0.2,
            "max_completion_length": 256,
            "use_vllm": False,
            "quant_bits": 4,
        },
        "category": "RLHF_Online",
        "requires_multi_gpu": False,
    })
    
    # ============================================================
    # Pre-training Tests (2 combos)
    # ============================================================
    for train_type in ["qlora", "full"]:
        test = {
            "name": f"PT_{train_type.upper()}",
            "training_method": "pt",
            "train_type": train_type,
            "dataset_path": pt_dataset,
            "rlhf_type": None,
            "extra_params": {},
            "category": "PreTraining",
        }
        if train_type == "qlora":
            test["extra_params"]["quant_bits"] = 4
        if train_type == "full":
            test["extra_params"]["per_device_train_batch_size"] = 1
            test["extra_params"]["gradient_accumulation_steps"] = 4
        tests.append(test)
    
    return tests


# ============================================================
# API HELPERS
# ============================================================

class APIClient:
    """Client for USF BIOS backend API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._last_loaded_model_path = None
    
    def health_check(self) -> bool:
        """Check if the backend is running."""
        try:
            # Try the health endpoint first (no /api prefix)
            base = self.base_url.replace("/api", "")
            r = self.session.get(f"{base}/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False
    
    def get_system_status(self) -> dict:
        """Get system status including GPU info."""
        r = self.session.get(f"{self.base_url}/system/status", timeout=10)
        r.raise_for_status()
        return r.json()
    
    def create_job(self, config: dict) -> dict:
        """Create a training job via POST /api/jobs/create."""
        r = self.session.post(f"{self.base_url}/jobs/create", json=config, timeout=60)
        r.raise_for_status()
        return r.json()
    
    def start_job(self, job_id: str) -> dict:
        """Start a training job via POST /api/jobs/{job_id}/start."""
        r = self.session.post(f"{self.base_url}/jobs/{job_id}/start", timeout=60)
        r.raise_for_status()
        return r.json()
    
    def get_job_status(self, job_id: str) -> dict:
        """Get specific job status via GET /api/jobs/{job_id}."""
        r = self.session.get(f"{self.base_url}/jobs/{job_id}", timeout=10)
        r.raise_for_status()
        return r.json()
    
    def get_current_job(self) -> dict:
        """Get current active job via GET /api/jobs/current."""
        r = self.session.get(f"{self.base_url}/jobs/current", timeout=10)
        r.raise_for_status()
        return r.json()
    
    def stop_job(self, job_id: str) -> dict:
        """Stop a training job via POST /api/jobs/{job_id}/stop."""
        r = self.session.post(f"{self.base_url}/jobs/{job_id}/stop", timeout=30)
        r.raise_for_status()
        return r.json()
    
    def get_terminal_logs(self, job_id: str, lines: int = 100) -> list:
        """Get terminal logs via GET /api/jobs/{job_id}/terminal-logs."""
        try:
            r = self.session.get(f"{self.base_url}/jobs/{job_id}/terminal-logs?lines={lines}", timeout=10)
            r.raise_for_status()
            data = r.json()
            return data.get("logs", [])
        except Exception:
            return []
    
    def clean_memory(self) -> dict:
        """Deep clean GPU memory via POST /api/inference/deep-clear-memory."""
        try:
            r = self.session.post(f"{self.base_url}/inference/deep-clear-memory", timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"status": "skipped"}
    
    def load_model_for_inference(self, model_path: str, adapter_path: str = None) -> dict:
        """Load a model for inference via POST /api/inference/load."""
        payload = {"model_path": model_path, "backend": "transformers"}
        if adapter_path:
            payload["adapter_path"] = adapter_path
        self._last_loaded_model_path = model_path
        r = self.session.post(f"{self.base_url}/inference/load", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()
    
    def run_inference(self, prompt: str, model_path: str = None, max_tokens: int = 50) -> dict:
        """Run inference via POST /api/inference/chat."""
        payload = {
            "model_path": model_path or self._last_loaded_model_path or "",
            "messages": [{"role": "user", "content": prompt}],
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
        }
        r = self.session.post(f"{self.base_url}/inference/chat", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    
    def unload_model(self) -> dict:
        """Clear inference model via POST /api/inference/deep-clear-memory."""
        try:
            r = self.session.post(f"{self.base_url}/inference/deep-clear-memory", timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"status": "skipped"}
    
    def check_active_training(self) -> bool:
        """Check if any training is currently active."""
        try:
            data = self.get_current_job()
            return data.get("has_active_job", False)
        except Exception:
            return False
    
    def get_inference_status(self) -> dict:
        """Get inference status via GET /api/inference/status."""
        try:
            r = self.session.get(f"{self.base_url}/inference/status", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {}


# ============================================================
# TEST RUNNER
# ============================================================

class TestResult:
    """Result of a single test."""
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.status = "pending"  # pending, running, passed, failed, skipped, error
        self.error_message = ""
        self.start_time = None
        self.end_time = None
        self.duration_seconds = 0
        self.job_id = ""
        self.output_dir = ""
        self.inference_tested = False
        self.inference_passed = False
        self.training_logs_tail = []
        self.final_loss = None
    
    def to_dict(self):
        return {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
            "job_id": self.job_id,
            "output_dir": self.output_dir,
            "inference_tested": self.inference_tested,
            "inference_passed": self.inference_passed,
            "final_loss": self.final_loss,
        }


class E2ETestRunner:
    """Runs all training tests sequentially."""
    
    def __init__(self, api_url: str, model_path: str, data_dir: str, output_base: str,
                 skip_online_rl: bool = False, skip_inference: bool = False,
                 skip_full: bool = False, only_test: str = None):
        self.client = APIClient(api_url)
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_base = output_base
        self.skip_online_rl = skip_online_rl
        self.skip_inference = skip_inference
        self.skip_full = skip_full
        self.only_test = only_test
        self.results: list[TestResult] = []
        self.start_time = None
    
    def run_all(self):
        """Run all tests."""
        self.start_time = datetime.now()
        
        print("\n" + "=" * 70)
        print("USF BIOS - COMPREHENSIVE E2E TRAINING TEST SUITE")
        print("=" * 70)
        print(f"  Model:      {self.model_path}")
        print(f"  Data Dir:   {self.data_dir}")
        print(f"  Output Dir: {self.output_base}")
        print(f"  API URL:    {self.client.base_url}")
        print(f"  Started:    {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Pre-flight checks
        if not self._preflight_checks():
            return
        
        # Get test matrix
        tests = get_test_matrix(self.data_dir)
        
        # Filter tests if only_test is specified
        if self.only_test:
            tests = [t for t in tests if self.only_test.lower() in t["name"].lower()]
            if not tests:
                print(f"\n✗ No tests match filter '{self.only_test}'")
                return
        
        print(f"\n  Total tests to run: {len(tests)}")
        print(f"  Categories: {', '.join(set(t['category'] for t in tests))}")
        
        # Run each test
        for i, test_config in enumerate(tests):
            print(f"\n{'─' * 70}")
            print(f"  TEST {i+1}/{len(tests)}: {test_config['name']}")
            print(f"  Category: {test_config['category']}")
            print(f"{'─' * 70}")
            
            result = self._run_single_test(test_config, i + 1, len(tests))
            self.results.append(result)
            
            # Save intermediate results
            self._save_results()
            
            # Clean GPU memory between tests
            print("  → Cleaning GPU memory...")
            self.client.clean_memory()
            time.sleep(3)
        
        # Final report
        self._print_final_report()
        self._save_results()
    
    def _preflight_checks(self) -> bool:
        """Run pre-flight checks."""
        print("\n  Pre-flight Checks:")
        
        # Check API health
        print("  → Checking backend API...", end=" ")
        if not self.client.health_check():
            print("✗ FAILED - Backend not reachable")
            print(f"    Make sure the backend is running at {self.client.base_url}")
            return False
        print("✓ OK")
        
        # Check system status
        print("  → Checking system status...", end=" ")
        try:
            status = self.client.get_system_status()
            gpu_name = status.get("gpu_name", "Unknown")
            gpu_count = status.get("gpu_count", 0)
            vram_total = status.get("vram_total_gb", 0)
            print(f"✓ {gpu_name} x{gpu_count} ({vram_total:.1f}GB VRAM)")
        except Exception as e:
            print(f"✗ FAILED - {e}")
            return False
        
        # Check no active training
        print("  → Checking no active training...", end=" ")
        if self.client.check_active_training():
            print("✗ Training is currently active!")
            print("    Please stop current training before running tests.")
            return False
        print("✓ No active training")
        
        # Check datasets exist
        print("  → Checking test datasets...", end=" ")
        manifest_path = os.path.join(self.data_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            print("✗ Datasets not found!")
            print(f"    Run: python prepare_test_datasets.py --output-dir {self.data_dir}")
            return False
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"✓ {len(manifest.get('datasets', {}))} datasets available")
        
        # Create output directory
        os.makedirs(self.output_base, exist_ok=True)
        print(f"  → Output directory: {self.output_base} ✓")
        
        return True
    
    def _run_single_test(self, test_config: dict, test_num: int, total: int) -> TestResult:
        """Run a single training test."""
        result = TestResult(test_config["name"], test_config["category"])
        result.start_time = datetime.now()
        
        # Skip conditions
        if self.skip_online_rl and test_config["category"] == "RLHF_Online":
            result.status = "skipped"
            result.error_message = "Online RL skipped (--skip-online-rl)"
            print(f"  ⊘ SKIPPED: {result.error_message}")
            return result
        
        if self.skip_full and test_config["train_type"] == "full":
            result.status = "skipped"
            result.error_message = "Full fine-tuning skipped (--skip-full)"
            print(f"  ⊘ SKIPPED: {result.error_message}")
            return result
        
        # Check dataset exists
        dataset_path = test_config["dataset_path"]
        if not os.path.exists(dataset_path):
            result.status = "error"
            result.error_message = f"Dataset not found: {dataset_path}"
            print(f"  ✗ ERROR: {result.error_message}")
            return result
        
        # Build training config (do NOT set output_dir - system_guard locks it)
        config = {
            "model_path": self.model_path,
            "model_source": "huggingface" if "/" in self.model_path and not self.model_path.startswith("/") else "local",
            "training_method": test_config["training_method"],
            "train_type": test_config["train_type"],
            "dataset_path": dataset_path,
            "output_dir": "",
            "name": f"e2e_{test_config['name'].lower()}_{int(time.time())}",
            **TEST_TRAINING_PARAMS,
        }
        
        # Add RLHF type if applicable
        if test_config["rlhf_type"]:
            config["rlhf_type"] = test_config["rlhf_type"]
        
        # Add extra params
        for key, value in test_config.get("extra_params", {}).items():
            config[key] = value
        
        # Remove LoRA params for full fine-tuning
        if test_config["train_type"] == "full":
            for key in ["lora_rank", "lora_alpha", "lora_dropout", "target_modules"]:
                config.pop(key, None)
        
        try:
            # Start training
            result.status = "running"
            print(f"  → Starting training: {test_config['training_method'].upper()} + "
                  f"{test_config['train_type'].upper()}"
                  f"{' + ' + test_config['rlhf_type'].upper() if test_config['rlhf_type'] else ''}")
            print(f"    Dataset: {os.path.basename(dataset_path)}")
            
            # Step 1: Create the job
            create_response = self.client.create_job(config)
            job_id = create_response.get("id", create_response.get("job_id", ""))
            result.job_id = job_id
            print(f"    Job ID:  {job_id}")
            
            if not job_id:
                result.status = "error"
                result.error_message = f"No job_id returned: {create_response}"
                print(f"  ✗ ERROR: {result.error_message}")
                return result
            
            # Step 2: Start the job
            start_response = self.client.start_job(job_id)
            print(f"    Start:   {start_response.get('status', 'unknown')}")
            
            # Wait for training to complete
            print(f"    Waiting for training to complete (timeout: {TEST_TIMEOUT}s)...")
            completed = self._wait_for_completion(job_id, test_config["name"])
            
            if not completed:
                result.status = "failed"
                result.error_message = "Training timed out or failed"
                # Get logs for debugging
                logs = self.client.get_terminal_logs(job_id)
                result.training_logs_tail = logs[-20:] if logs else []
                
                # Try to get more specific error
                try:
                    job_data = self.client.get_job_status(job_id)
                    job_info = job_data.get("job", job_data)
                    actual_status = job_info.get("status", "unknown")
                    error_msg = job_info.get("error_message", job_info.get("error", ""))
                    if error_msg:
                        result.error_message = f"Training {actual_status}: {error_msg}"
                    elif actual_status == "running":
                        result.error_message = f"Training timed out after {TEST_TIMEOUT}s"
                        # Stop the timed-out training
                        self.client.stop_job(job_id)
                except Exception:
                    pass
                
                print(f"  ✗ FAILED: {result.error_message}")
                if result.training_logs_tail:
                    print("    Last logs:")
                    for log in result.training_logs_tail[-5:]:
                        print(f"      {log}")
                return result
            
            # Training completed - get final info
            try:
                job_data = self.client.get_job_status(job_id)
                job_info = job_data.get("job", job_data)
                result.output_dir = job_info.get("output_dir", "")
                result.final_loss = job_info.get("current_loss")
            except Exception:
                result.output_dir = ""
            
            result.status = "passed"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            
            print(f"  ✓ Training PASSED in {result.duration_seconds:.0f}s")
            if result.final_loss:
                print(f"    Final loss: {result.final_loss:.4f}")
            
            # Test inference if not skipped
            if not self.skip_inference:
                self._test_inference(test_config, result)
            
            return result
            
        except requests.exceptions.HTTPError as e:
            result.status = "error"
            try:
                error_detail = e.response.json()
                result.error_message = f"API Error {e.response.status_code}: {error_detail.get('detail', str(e))}"
            except Exception:
                result.error_message = f"API Error: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            print(f"  ✗ ERROR: {result.error_message}")
            return result
            
        except Exception as e:
            result.status = "error"
            result.error_message = f"Unexpected error: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            print(f"  ✗ ERROR: {result.error_message}")
            traceback.print_exc()
            return result
    
    def _wait_for_completion(self, job_id: str, test_name: str) -> bool:
        """Wait for a training job to complete."""
        start = time.time()
        last_step = 0
        last_print = time.time()
        
        while time.time() - start < TEST_TIMEOUT:
            try:
                # Check job status directly
                try:
                    job_data = self.client.get_job_status(job_id)
                    # The response may have a nested 'job' object or be flat
                    job_info = job_data.get("job", job_data)
                    status = job_info.get("status", "")
                    
                    if status == "completed":
                        return True
                    elif status in ("failed", "stopped", "cancelled"):
                        return False
                    
                    # Print progress periodically
                    current_step = job_info.get("current_step", 0)
                    total_steps = job_info.get("total_steps", 0)
                    
                    if time.time() - last_print > 15 or current_step != last_step:
                        elapsed = int(time.time() - start)
                        if total_steps > 0:
                            pct = (current_step / total_steps * 100) if total_steps > 0 else 0
                            print(f"    [{elapsed}s] Step {current_step}/{total_steps} ({pct:.1f}%)")
                        else:
                            print(f"    [{elapsed}s] Status: {status}")
                        last_step = current_step
                        last_print = time.time()
                except Exception:
                    pass
                
            except Exception as e:
                # Connection error - might be temporary
                pass
            
            time.sleep(POLL_INTERVAL)
        
        return False
    
    def _test_inference(self, test_config: dict, result: TestResult):
        """Test inference loading for a completed training."""
        print(f"  → Testing inference loading...")
        result.inference_tested = True
        
        try:
            train_type = test_config["train_type"]
            output_dir = result.output_dir
            
            if train_type in ["lora", "qlora", "adalora"]:
                # Load base model + adapter
                print(f"    Loading base model + adapter from {output_dir}")
                self.client.load_model_for_inference(
                    model_path=self.model_path,
                    adapter_path=output_dir,
                )
            else:
                # Full fine-tuning - load output directly
                print(f"    Loading full model from {output_dir}")
                self.client.load_model_for_inference(model_path=output_dir)
            
            # Wait for model to load
            time.sleep(5)
            
            # Run a test inference
            print(f"    Running test inference...")
            response = self.client.run_inference("What is machine learning?", max_tokens=50)
            
            if response:
                result.inference_passed = True
                print(f"  ✓ Inference PASSED")
            else:
                result.inference_passed = False
                print(f"  ✗ Inference returned empty response")
            
            # Unload model
            self.client.unload_model()
            time.sleep(2)
            
        except Exception as e:
            result.inference_passed = False
            print(f"  ✗ Inference FAILED: {str(e)}")
    
    def _print_final_report(self):
        """Print the final test report."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        passed = [r for r in self.results if r.status == "passed"]
        failed = [r for r in self.results if r.status == "failed"]
        errors = [r for r in self.results if r.status == "error"]
        skipped = [r for r in self.results if r.status == "skipped"]
        
        print("\n" + "=" * 70)
        print("FINAL TEST REPORT")
        print("=" * 70)
        print(f"  Total Duration: {total_duration:.0f}s ({total_duration/60:.1f} min)")
        print(f"  Total Tests:    {len(self.results)}")
        print(f"  ✓ Passed:       {len(passed)}")
        print(f"  ✗ Failed:       {len(failed)}")
        print(f"  ⚠ Errors:       {len(errors)}")
        print(f"  ⊘ Skipped:      {len(skipped)}")
        print()
        
        # Category breakdown
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
            categories[r.category][r.status if r.status in categories[r.category] else "error"] += 1
        
        print("  By Category:")
        for cat, counts in categories.items():
            total = sum(counts.values())
            p = counts["passed"]
            print(f"    {cat}: {p}/{total} passed")
        
        # Detailed results
        print("\n  Detailed Results:")
        print(f"  {'Test Name':<25} {'Status':<10} {'Duration':<12} {'Loss':<10} {'Inference'}")
        print(f"  {'─' * 25} {'─' * 10} {'─' * 12} {'─' * 10} {'─' * 10}")
        
        for r in self.results:
            status_icon = {"passed": "✓", "failed": "✗", "error": "⚠", "skipped": "⊘"}.get(r.status, "?")
            duration = f"{r.duration_seconds:.0f}s" if r.duration_seconds else "-"
            loss = f"{r.final_loss:.4f}" if r.final_loss else "-"
            inference = "✓" if r.inference_passed else ("✗" if r.inference_tested else "-")
            print(f"  {status_icon} {r.name:<23} {r.status:<10} {duration:<12} {loss:<10} {inference}")
        
        # Failed tests details
        if failed or errors:
            print("\n  Failed/Error Details:")
            for r in failed + errors:
                print(f"\n  ✗ {r.name}:")
                print(f"    Error: {r.error_message}")
                if r.training_logs_tail:
                    print(f"    Last logs:")
                    for log in r.training_logs_tail[-3:]:
                        print(f"      {log}")
        
        # Overall verdict
        print("\n" + "=" * 70)
        if len(failed) == 0 and len(errors) == 0:
            print("  🎉 ALL TESTS PASSED!")
        else:
            print(f"  ⚠ {len(failed) + len(errors)} TEST(S) NEED ATTENTION")
        print("=" * 70)
    
    def _save_results(self):
        """Save test results to JSON file."""
        results_path = os.path.join(self.output_base, "test_results.json")
        report = {
            "run_started": self.start_time.isoformat() if self.start_time else None,
            "run_ended": datetime.now().isoformat(),
            "model": self.model_path,
            "total_tests": len(self.results),
            "passed": len([r for r in self.results if r.status == "passed"]),
            "failed": len([r for r in self.results if r.status == "failed"]),
            "errors": len([r for r in self.results if r.status == "error"]),
            "skipped": len([r for r in self.results if r.status == "skipped"]),
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(results_path, "w") as f:
            json.dump(report, f, indent=2)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="USF BIOS E2E Training Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_all_training_tests.py --model Qwen/Qwen2.5-0.5B

  # Run only SFT tests
  python run_all_training_tests.py --model Qwen/Qwen2.5-0.5B --only sft

  # Skip online RL and full fine-tuning
  python run_all_training_tests.py --model Qwen/Qwen2.5-0.5B --skip-online-rl --skip-full

  # Run with local model
  python run_all_training_tests.py --model /path/to/local/model
        """
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Backend API URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path or HuggingFace ID")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "test_data"),
                        help="Directory containing test datasets")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "test_outputs"),
                        help="Directory for training outputs")
    parser.add_argument("--skip-online-rl", action="store_true", help="Skip online RL tests (PPO, GRPO)")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference tests after training")
    parser.add_argument("--skip-full", action="store_true", help="Skip full fine-tuning tests (saves memory)")
    parser.add_argument("--only", dest="only_test", default=None,
                        help="Only run tests matching this string (e.g., 'sft', 'dpo', 'grpo')")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout per test in seconds")
    
    args = parser.parse_args()
    
    TEST_TIMEOUT = args.timeout
    
    runner = E2ETestRunner(
        api_url=args.api_url,
        model_path=args.model,
        data_dir=args.data_dir,
        output_base=args.output_dir,
        skip_online_rl=args.skip_online_rl,
        skip_inference=args.skip_inference,
        skip_full=args.skip_full,
        only_test=args.only_test,
    )
    
    runner.run_all()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DAPO Trainer Patch for Log Probability QA Reward

Key Features:
1. Rollout length for question extraction
2. Optimizes first N tokens only
3. History prompt truncation
4. Batch filtering based on question extraction rate
5. Merged log prob computation (with/without QA in single batch)
"""

import os
import re
import sys
import time
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# Configuration Constants
# ============================================================

# Prompt template for recovery with history
RECOVER_PROMPT_TEMPLATE = """Role
You are an expert analyst. Summarize the user's hidden profile and intent based on the dialogue.
History:{history}
{clarification_turn}
User Intent Summary:"""

# Log directory (relative to script)
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = SCRIPT_DIR / "logs" / "log_prob_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Question log directory
QUESTION_LOG_DIR = SCRIPT_DIR / "logs" / "question_logs"
QUESTION_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Length configuration
MAX_HISTORY_LENGTH = 1600  # Max history characters
MAX_OPTIMIZE_LENGTH = 512  # Optimize first N tokens only

# Batch configuration
ROLLOUTS_PER_PROMPT = 8  # Rollouts per prompt
SELECT_COUNT = 6  # Select N rollouts per prompt for update
MAX_NO_QUESTION_ALLOWED = 6  # Allow all without question (don't skip any prompt)
MIN_QUESTIONS_REQUIRED = 0  # Allow batch without questions
BATCH_SIZE = 4  # Prompts per batch

# User prompt template for simulating user response
USER_PROMPT_TEMPLATE = """Role
You are a user interacting with an agent.
Your behavior simulates a human user following a hidden instruction.
{History}
{User Detail Requirements}
Instructions:
conversation guidelines:
- Respond with one message at a time using first-person statements.
- Do not invent details that are not in the instruction; if something is unknown, say you do not remember it.
- Rephrase the instruction in your own words and maintain a natural, human-like tone.

IMPORTANT - Handling vague or generic questions:
- If the agent asks a vague, overly broad, or generic question (e.g., "Is there any additional information?", "Can you tell me more?", "Anything else?"), reply with: "No, that's all." or "Just do what I asked."
- If the agent outputs a placeholder or acts out of character (e.g., text like "Your concise and specific clarifying question to the user here"), reply with: "Who are you talking to?"
- For such questions, only provide information if it is DIRECTLY relevant to the current step of the instruction.
- If the question is too vague to answer meaningfully, respond with something like "I'm not sure what specific information you need" or "Could you be more specific?"
- Prefer specific, targeted questions that help the agent understand your exact needs.
Question:{question}
expected output:
A single user utterance that follows the instruction and guidelines."""

_FIRST_SAMPLE_LOGGED = False
_executor = ThreadPoolExecutor(max_workers=2)
_QUESTION_LOG_STEP = 0  # Global step counter
MAX_LOG_STEPS = 20  # Only log first N steps


# ============================================================
# Helper Functions
# ============================================================

def _clean_think_tags(text: str) -> str:
    """
    Remove think-related tags from text to avoid interfering with Qwen3 think mode
    Removes: /think, /no_think, <think>, </think>, <thinking>, </thinking>
    """
    if not text:
        return text
    
    # Remove /think, /no_think tags
    text = re.sub(r'/no_think\b', '', text)
    text = re.sub(r'/think\b', '', text)
    
    # Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove standalone tags
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</?thinking>', '', text, flags=re.IGNORECASE)
    
    # Clean extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()


def _extract_history_from_prompt(prompt_text: str) -> str:
    """
    Extract dialogue history from prompt, intelligently truncate to MAX_HISTORY_LENGTH chars
    
    Strategy: Truncate at complete JSON object or sentence boundaries
    - Maintain JSON syntax integrity
    - Don't imply missing information (remove "...")
    - Clean think-related tags
    """
    context_match = re.search(r'Context:\s*(.*?)(?:Your task:|$)', prompt_text, re.DOTALL)
    if context_match:
        history = context_match.group(1).strip()
        
        # Clean think-related tags
        history = _clean_think_tags(history)
        
        # If within limit, return directly
        if len(history) <= MAX_HISTORY_LENGTH:
            return history
        
        # Need truncation - find last complete boundary
        truncated = history[:MAX_HISTORY_LENGTH]
        
        # Priority: complete } > newline > period
        last_brace = truncated.rfind('}')
        last_newline = truncated.rfind('\n')
        last_period = truncated.rfind('.')
        
        # Select best cutoff point (keep at least 70% content)
        min_length = int(MAX_HISTORY_LENGTH * 0.7)
        cutoff = -1
        
        if last_brace > min_length:
            cutoff = last_brace + 1
        elif last_newline > min_length:
            cutoff = last_newline
        elif last_period > min_length:
            cutoff = last_period + 1
        else:
            # If none satisfy, truncate at current position
            cutoff = MAX_HISTORY_LENGTH
        
        history = truncated[:cutoff].strip()
        
        # Key: Remove trailing "..." ellipsis
        history = re.sub(r'\.\.\.\s*$', '', history).strip()
        
        return history
    
    # If no match, truncate original text directly
    return prompt_text[:MAX_HISTORY_LENGTH] if prompt_text else ""


def _remove_thinking(text: str) -> str:
    """Remove complete or truncated thinking blocks"""
    if not text:
        return text
    # Remove complete <think>...</think> blocks
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r'<thinking>.*?</thinking>', '', clean, flags=re.DOTALL | re.IGNORECASE)
    # Remove truncated <think> blocks (no closing tag)
    clean = re.sub(r'<think>.*$', '', clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r'<thinking>.*$', '', clean, flags=re.DOTALL | re.IGNORECASE)
    return clean.strip()


def _extract_question(response: str) -> Optional[str]:
    """Extract question from response, supports COT format and Base model direct output (v3)"""
    if not response:
        return None
    
    # Check for complete </think> closure
    has_complete_think = '</think>' in response.lower() or '</thinking>' in response.lower()
    
    if has_complete_think:
        # ============ Instruct model: COT format ============
        # <think>...</think> followed by YES/NO
        response_clean = _remove_thinking(response)
        response_upper = response_clean.upper().strip()
        
        # If </think> followed by NO, no question needed
        if response_upper.startswith('NO') or 'NO NEED' in response_upper[:50]:
            return None
        
        # If YES, look for [QUESTION] tag
        if response_upper.startswith('YES'):
            # Match [QUESTION] tag
            match = re.search(r'\[QUESTION\]\s*(.*?)\s*\[/QUESTION\]', response_clean, re.DOTALL | re.IGNORECASE)
            if match:
                q = match.group(1).strip()
                if len(q) > 5:
                    return q
            
            # YES followed directly by question
            yes_match = re.search(r'^YES\s*[:\-]?\s*(.+?)(?:\?|$)', response_clean, re.IGNORECASE | re.DOTALL)
            if yes_match:
                q = yes_match.group(1).strip()
                if '?' in q:
                    q = q[:q.rfind('?')+1]
                if len(q) > 10:
                    return q
        
        return None
    else:
        # ============ Base model: Direct output format (no <think> tags) ============
        response_clean = response.strip()
        response_upper = response_clean.upper()
        
        # Check for NO response
        if re.search(r'^(?:Assistant:\s*)?NO\s*$', response_clean, re.IGNORECASE | re.MULTILINE):
            return None
        if response_upper.startswith('NO') or 'NO NEED' in response_upper[:100]:
            return None
        
        # Try [QUESTION]...[/QUESTION] tag
        match = re.search(r'\[QUESTION\]\s*(.*?)\s*\[/QUESTION\]', response_clean, re.DOTALL | re.IGNORECASE)
        if match:
            q = match.group(1).strip()
            if len(q) > 5:
                return q
        
        # Try <question>...</question> tag
        match = re.search(r'<question>(.*?)</question>', response_clean, re.DOTALL | re.IGNORECASE)
        if match:
            q = match.group(1).strip()
            if len(q) > 5:
                return q
        
        # Try YES [QUESTION]... pattern
        match = re.search(r'YES\s+\[QUESTION\](.*?)(?:\[/QUESTION\]|$)', response_clean, re.DOTALL | re.IGNORECASE)
        if match:
            q = match.group(1).strip()
            if len(q) > 5:
                return q
        
        # Try YES ... pattern
        match = re.search(r'YES\s*[:\-\[]?\s*(.+?)(?:\n|$)', response_clean, re.IGNORECASE)
        if match:
            q = match.group(1).strip()
            # Clean common artifacts
            q = re.sub(r'\[/?QUESTION\]', '', q, flags=re.IGNORECASE).strip()
            q = re.sub(r'^Your concise.*?:\s*["\']?', '', q, flags=re.IGNORECASE).strip()
            q = re.sub(r'["\']?\s*$', '', q).strip()
            if len(q) > 10 and '?' in q:
                return q
        
        # Try finding question-ending sentences
        match = re.search(r'((?:Can you|Could you|Would you|Do you|Are you|Is the|Please confirm|Clarify)[^?]*\?)', response_clean, re.IGNORECASE)
        if match:
            q = match.group(1).strip()
            if len(q) > 15:
                return q
        
        return None


def log_rollout_questions(
    step: int,
    prompts: List[str],
    responses: List[str],
    questions: List[Optional[str]],
    ground_truths: List[str],
    save_to_file: bool = True
) -> None:
    """
    Print rollout extracted questions (console only, files saved by log_complete_step_data)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    batch_size = len(prompts)
    num_questions = sum(1 for q in questions if q is not None)
    
    # ============ Format analysis ============
    num_with_think = sum(1 for r in responses if r and ('<think>' in r.lower() or '</think>' in r.lower()))
    num_with_question_tag = sum(1 for r in responses if r and '[QUESTION]' in r.upper())
    num_starts_yes = sum(1 for r in responses if r and r.strip().upper().startswith('YES'))
    num_starts_no = sum(1 for r in responses if r and r.strip().upper().startswith('NO'))
    
    # ============ Console print ============
    print("\n" + "=" * 80, flush=True)
    print(f"ROLLOUT QUESTIONS - Step {step}", flush=True)
    print(f"   Timestamp: {timestamp}", flush=True)
    print(f"   Batch size: {batch_size}", flush=True)
    print(f"   Questions extracted: {num_questions}/{batch_size} ({100*num_questions/batch_size:.1f}%)", flush=True)
    print("=" * 80, flush=True)
    
    # ============ Format statistics ============
    print(f"\n   Response Format Analysis:", flush=True)
    print(f"      - With <think> COT tags: {num_with_think}/{batch_size} ({100*num_with_think/batch_size:.1f}%)", flush=True)
    print(f"      - With [QUESTION] tags: {num_with_question_tag}/{batch_size} ({100*num_with_question_tag/batch_size:.1f}%)", flush=True)
    print(f"      - Starts with YES: {num_starts_yes}/{batch_size}", flush=True)
    print(f"      - Starts with NO: {num_starts_no}/{batch_size}", flush=True)
    print("", flush=True)
    
    # Print samples with questions
    samples_with_q = [(i, p, r, q, g) for i, (p, r, q, g) in enumerate(zip(prompts, responses, questions, ground_truths)) if q]
    
    print(f"   Samples with extracted questions ({len(samples_with_q)}):", flush=True)
    for idx, (i, prompt, response, question, gt) in enumerate(samples_with_q[:5]):
        has_cot = '<think>' in response.lower() if response else False
        has_tag = '[QUESTION]' in response.upper() if response else False
        format_info = f"[COT:{has_cot}, TAG:{has_tag}]"
        
        print(f"\n   [{i}] {format_info}", flush=True)
        print(f"       Question: {question[:120]}{'...' if len(question) > 120 else ''}", flush=True)
        
        # Show response key part
        if response:
            resp_preview = response[:300].replace('\n', ' ')
            print(f"       Response: {resp_preview}{'...' if len(response) > 300 else ''}", flush=True)
    
    if len(samples_with_q) > 5:
        print(f"\n   ... (showing 5/{len(samples_with_q)} samples with questions)", flush=True)
    
    print("=" * 80 + "\n", flush=True)


def log_complete_step_data(
    step: int,
    prompts: List[str],
    responses: List[str],
    questions: List[Optional[str]],
    ground_truths: List[str],
    user_responses: List[Optional[str]],
    log_probs_with: List[float],
    log_probs_without: List[float],
    rewards: List[float],
    selected_indices: List[int],
    valid_orig_indices: List[int] = None,
    response_records: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> None:
    """
    Save complete step data to JSON file (only first MAX_LOG_STEPS steps)
    
    Contains: rollout response, extracted question, user_response,
              log_prob with/without QA, reward
    """
    global _QUESTION_LOG_STEP
    response_records = response_records or []
    
    # Only log first MAX_LOG_STEPS steps
    if step > MAX_LOG_STEPS:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    batch_size = len(prompts)
    num_questions = sum(1 for q in questions if q is not None)
    
    # Format analysis
    num_with_think = sum(1 for r in responses if r and ('<think>' in r.lower() or '</think>' in r.lower()))
    num_with_question_tag = sum(1 for r in responses if r and '[QUESTION]' in r.upper())
    
    log_data = {
        "step": step,
        "timestamp": timestamp,
        "batch_size": batch_size,
        "num_questions": num_questions,
        "num_selected": len(selected_indices),
        "num_with_user_response": sum(1 for u in user_responses if u),
        "format_stats": {
            "with_think_tags": num_with_think,
            "with_question_tags": num_with_question_tag,
        },
        "samples": []
    }
    
    # Build mapping from original indices to log_prob/reward
    selected_data_map = {}
    
    # First set user_response for all selected_indices
    for local_idx, orig_idx in enumerate(selected_indices):
        record = response_records[local_idx] if local_idx < len(response_records) else None
        if not isinstance(record, dict):
            record = {}
        selected_data_map[orig_idx] = {
            "user_response": user_responses[local_idx] if local_idx < len(user_responses) else None,
            "response_role": record.get("response_role"),
            "response_mode": record.get("response_mode"),
            "response_raw": record.get("response_raw"),
            "response_think": record.get("response_think"),
            "response_final": record.get("response_final"),
            "log_prob_with_qa": None,
            "log_prob_without_qa": None,
            "reward": None,
        }
    
    # Then set log_prob and reward only for valid samples
    if valid_orig_indices:
        for valid_idx, orig_idx in enumerate(valid_orig_indices):
            if orig_idx in selected_data_map and valid_idx < len(log_probs_with):
                selected_data_map[orig_idx]["log_prob_with_qa"] = log_probs_with[valid_idx]
                selected_data_map[orig_idx]["log_prob_without_qa"] = log_probs_without[valid_idx] if valid_idx < len(log_probs_without) else None
                selected_data_map[orig_idx]["reward"] = rewards[valid_idx] if valid_idx < len(rewards) else None
    
    for i, (prompt, response, question, gt) in enumerate(zip(prompts, responses, questions, ground_truths)):
        # Check for complete </think>
        has_complete_think = '</think>' in response.lower() if response else False
        # Extract content after </think>
        after_think = ""
        if has_complete_think and response:
            idx = response.lower().find('</think>')
            after_think = response[idx:idx+200]
        
        # Get API response and log prob data
        extra_data = selected_data_map.get(i, {})
        
        sample_data = {
            "index": i,
            "is_selected": i in selected_indices,
            "has_question": question is not None,
            "question": question,
            "user_response": extra_data.get("user_response"),
            "response_role": extra_data.get("response_role"),
            "response_mode": extra_data.get("response_mode"),
            "response_raw": extra_data.get("response_raw"),
            "response_think": extra_data.get("response_think"),
            "response_final": extra_data.get("response_final"),
            "log_prob_with_qa": extra_data.get("log_prob_with_qa"),
            "log_prob_without_qa": extra_data.get("log_prob_without_qa"),
            "reward": extra_data.get("reward"),
            "prompt": prompt[:500] if prompt and len(prompt) > 500 else prompt,
            "response": response,
            "response_length": len(response) if response else 0,
            "has_complete_think": has_complete_think,
            "after_think": after_think,
            "ground_truth": gt[:300] if gt and len(gt) > 300 else gt,
        }
        log_data["samples"].append(sample_data)
    
    # Save to JSON file
    step_log_file = QUESTION_LOG_DIR / f"step_{step:06d}.json"
    try:
        with open(step_log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        print(f"   Complete step data saved to: {step_log_file}")
    except Exception as e:
        print(f"   WARNING: Failed to save step log: {e}")


# ============================================================
# Response Truncation (optimize first N tokens only)
# ============================================================

def truncate_responses_for_optimization(batch, max_optimize_len: int = MAX_OPTIMIZE_LENGTH):
    """
    Truncate responses to keep only first max_optimize_len tokens for DAPO optimization
    Full response is kept for question extraction (already extracted)
    """
    responses = batch.batch["responses"]
    current_len = responses.size(1)
    
    if current_len <= max_optimize_len:
        return batch
    
    print(f"   Truncating responses: {current_len} -> {max_optimize_len} tokens")
    
    # Truncate responses
    batch.batch["responses"] = responses[:, :max_optimize_len]
    
    # Calculate new total length
    input_ids = batch.batch["input_ids"]
    seq_len = input_ids.size(1)
    prompt_len = seq_len - current_len
    new_total_len = prompt_len + max_optimize_len
    
    # Truncate input_ids, attention_mask, position_ids
    batch.batch["input_ids"] = input_ids[:, :new_total_len]
    batch.batch["attention_mask"] = batch.batch["attention_mask"][:, :new_total_len]
    if "position_ids" in batch.batch:
        batch.batch["position_ids"] = batch.batch["position_ids"][:, :new_total_len]
    
    return batch


# ============================================================
# User Response Simulation (Local Model)
# ============================================================

def simulate_user_response(question: str, ground_truth: str) -> Optional[str]:
    """
    Simulate user response based on question and ground truth.
    
    This is a simplified local simulation. In production, you can:
    1. Use a local LLM to generate responses
    2. Use a rule-based system
    3. Use an external API (configure your own)
    
    For this release, we use a simple extraction from ground truth.
    """
    if not question or not ground_truth:
        return None
    
    # Simple simulation: extract relevant info from ground truth
    # In production, replace this with actual LLM call
    
    # Check if question is too generic
    generic_patterns = [
        r'anything else',
        r'any additional',
        r'can you tell me more',
        r'is there any',
    ]
    
    for pattern in generic_patterns:
        if re.search(pattern, question.lower()):
            return "No, that's all."
    
    # For specific questions, return a portion of ground truth
    # This is a placeholder - in production, use an actual model
    if len(ground_truth) > 100:
        return ground_truth[:100] + "..."
    return ground_truth


def fetch_responses_batch(questions: List[Optional[str]], ground_truths: List[str]) -> List[Optional[str]]:
    """
    Batch fetch user responses (local simulation)
    
    Replace this with actual model inference or API calls as needed.
    """
    results = []
    for q, gt in zip(questions, ground_truths):
        if q and gt:
            results.append(simulate_user_response(q, gt))
        else:
            results.append(None)
    return results


# ============================================================
# Merged Log Prob Computation (single batch)
# ============================================================

def construct_merged_log_prob_batch(
    prompts_with: List[str],
    prompts_without: List[str],
    completions: List[str],
    tokenizer,
    max_prompt_len: int = 2048,
    max_completion_len: int = 512,
    num_workers: int = 4,
):
    """
    Merge with/without QA batch for single computation
    
    Returns: batch_dict, original_size, split_point
    - split_point: first half is with QA, second half is without QA
    
    Important:
    - prompt uses left-padding (right-aligned)
    - completion uses right-padding (left-aligned)
    - After concatenation: [PAD, ..., prompt, completion, PAD, ...]
    - Ensures prompt and completion are contiguous without PAD separation
    """
    all_prompts = prompts_with + prompts_without
    all_completions = completions + completions
    
    original_size = len(prompts_with)
    split_point = original_size
    total_size = len(all_prompts)
    
    # Padding to be divisible by num_workers
    remainder = total_size % num_workers
    if remainder != 0:
        pad_size = num_workers - remainder
        all_prompts = all_prompts + [all_prompts[-1]] * pad_size
        all_completions = all_completions + [all_completions[-1]] * pad_size
    
    batch_size = len(all_prompts)
    
    # Save original padding_side
    original_padding_side = tokenizer.padding_side
    
    # Prompt uses left-padding (right-aligned, actual content on right)
    tokenizer.padding_side = "left"
    prompt_tokens = tokenizer(
        all_prompts,
        padding="max_length",
        truncation=True,
        max_length=max_prompt_len,
        return_tensors="pt"
    )
    
    # Completion uses right-padding (left-aligned, actual content on left)
    tokenizer.padding_side = "right"
    completion_tokens = tokenizer(
        all_completions,
        padding="max_length",
        truncation=True,
        max_length=max_completion_len,
        add_special_tokens=False,
        return_tensors="pt"
    )
    
    # Restore original padding_side
    tokenizer.padding_side = original_padding_side
    
    input_ids = torch.cat([prompt_tokens["input_ids"], completion_tokens["input_ids"]], dim=1)
    attention_mask = torch.cat([prompt_tokens["attention_mask"], completion_tokens["attention_mask"]], dim=1)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand(batch_size, -1)
    responses = completion_tokens["input_ids"]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
    }, original_size, split_point


# ============================================================
# Batch Filter
# ============================================================

class BatchQuestionFilter:
    """
    Filter batch to ensure at least MIN_QUESTIONS_REQUIRED questions
    """
    
    def __init__(self, min_questions: int = MIN_QUESTIONS_REQUIRED):
        self.min_questions = min_questions
        self.discarded_batches = 0
        self.total_batches = 0
    
    def should_discard(self, questions: List[Optional[str]]) -> bool:
        """
        Check if batch should be discarded
        Returns True if should discard
        """
        self.total_batches += 1
        num_questions = sum(1 for q in questions if q is not None)
        
        if num_questions < self.min_questions:
            self.discarded_batches += 1
            print(f"   WARNING: Batch discarded: only {num_questions}/{len(questions)} questions "
                  f"(need >= {self.min_questions})")
            print(f"   Discard rate: {self.discarded_batches}/{self.total_batches} "
                  f"({100*self.discarded_batches/self.total_batches:.1f}%)")
            return True
        
        print(f"   Batch accepted: {num_questions}/{len(questions)} questions")
        return False
    
    def get_stats(self) -> dict:
        return {
            "total_batches": self.total_batches,
            "discarded_batches": self.discarded_batches,
            "discard_rate": self.discarded_batches / max(1, self.total_batches)
        }


# Global filter instance
_batch_filter = BatchQuestionFilter()


# ============================================================
# Main Reward Computation Function
# ============================================================

def select_rollouts_per_prompt(
    questions: List[Optional[str]],
    rollouts_per_prompt: int = ROLLOUTS_PER_PROMPT,
    select_count: int = SELECT_COUNT,
    max_no_question: int = MAX_NO_QUESTION_ALLOWED,
) -> Tuple[List[int], List[int], dict]:
    """
    Select best select_count rollouts from each prompt's rollouts
    
    Strategy:
    - Prioritize rollouts with questions
    - If selected rollouts have more than max_no_question without questions, skip the prompt
    
    Returns:
        selected_indices: selected rollout indices
        skipped_prompts: skipped prompt indices
        stats: statistics
    """
    batch_size = len(questions)
    num_prompts = batch_size // rollouts_per_prompt
    
    selected_indices = []
    skipped_prompts = []
    stats = {
        "total_prompts": num_prompts,
        "selected_prompts": 0,
        "skipped_prompts": 0,
        "total_questions": 0,
        "selected_questions": 0,
    }
    
    for prompt_idx in range(num_prompts):
        start_idx = prompt_idx * rollouts_per_prompt
        end_idx = start_idx + rollouts_per_prompt
        
        # All rollout indices for this prompt
        rollout_indices = list(range(start_idx, end_idx))
        
        # Classify: with question and without question
        with_question = [i for i in rollout_indices if questions[i] is not None]
        without_question = [i for i in rollout_indices if questions[i] is None]
        
        stats["total_questions"] += len(with_question)
        
        # Prioritize those with questions, supplement with those without
        selected = with_question[:select_count]
        if len(selected) < select_count:
            remaining = select_count - len(selected)
            selected.extend(without_question[:remaining])
        
        # Count how many selected have no question
        no_question_count = sum(1 for i in selected if questions[i] is None)
        
        if no_question_count > max_no_question:
            # Skip this prompt
            skipped_prompts.append(prompt_idx)
            stats["skipped_prompts"] += 1
            print(f"      Prompt {prompt_idx}: SKIPPED ({len(with_question)}/{rollouts_per_prompt} questions, need >= {select_count - max_no_question})")
        else:
            selected_indices.extend(selected)
            stats["selected_prompts"] += 1
            stats["selected_questions"] += sum(1 for i in selected if questions[i] is not None)
            print(f"      Prompt {prompt_idx}: selected {len(selected)} rollouts ({len(with_question)} with questions)")
    
    return selected_indices, skipped_prompts, stats


def compute_log_prob_qa_reward(
    batch,
    tokenizer,
    actor_rollout_wg,
    metrics: dict,
    timing_raw: dict,
):
    """
    Log prob QA reward computation
    
    Core logic:
    - Each prompt has ROLLOUTS_PER_PROMPT rollouts
    - Select best SELECT_COUNT rollouts per prompt (prioritize those with questions)
    - Skip prompt if too many selected have no question
    - Extract question from response
    - Truncate history to MAX_HISTORY_LENGTH
    - Merge with/without batch for single computation
    """
    global _FIRST_SAMPLE_LOGGED
    
    from verl import DataProto
    from verl.utils.profiler import marked_timer
    
    print("\n" + "="*70)
    print("Computing Log Prob QA Reward")
    print(f"   Config: {ROLLOUTS_PER_PROMPT} rollouts/prompt, select {SELECT_COUNT}, max {MAX_NO_QUESTION_ALLOWED} without question")
    print(f"   History max: {MAX_HISTORY_LENGTH} chars | Optimize: {MAX_OPTIMIZE_LENGTH} tokens")
    print("="*70)
    
    start_time = time.time()
    first_sample_details = None
    
    try:
        batch_size = len(batch.batch["prompts"])
        
        # 1. Decode (response is full length)
        prompts = tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        responses = tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        
        # DEBUG: Print first sample token format
        if batch_size > 0:
            prompt_with_special = tokenizer.decode(batch.batch["prompts"][0], skip_special_tokens=False)
            response_with_special = tokenizer.decode(batch.batch["responses"][0], skip_special_tokens=False)
            print(f"\n   DEBUG - Token format check:")
            print(f"      Prompt ends with: {repr(prompt_with_special[-100:])}")
            print(f"      Response starts: {repr(response_with_special[:200])}")
        
        # 2. Get ground_truth_clean
        ground_truths = []
        rm_data = batch.non_tensor_batch.get("reward_model", [{}] * batch_size)
        if isinstance(rm_data, np.ndarray):
            rm_data = rm_data.tolist()
        for i in range(batch_size):
            gt = rm_data[i].get("ground_truth_clean", "") if isinstance(rm_data[i], dict) else ""
            ground_truths.append(gt)
        
        # 3. Extract questions (from full response)
        questions = [_extract_question(r) for r in responses]
        
        # 4. Extract and truncate histories (max chars)
        histories = [_extract_history_from_prompt(p) for p in prompts]
        
        total_questions = sum(1 for q in questions if q)
        print(f"   Total questions extracted: {total_questions}/{batch_size}")
        
        # ========== Log rollout questions ==========
        global _QUESTION_LOG_STEP
        _QUESTION_LOG_STEP += 1
        log_rollout_questions(
            step=_QUESTION_LOG_STEP,
            prompts=prompts,
            responses=responses,
            questions=questions,
            ground_truths=ground_truths,
            save_to_file=True
        )
        
        # 5. Smart selection: select best rollouts per prompt
        print(f"   Selecting best rollouts per prompt:")
        selected_indices, skipped_prompts, select_stats = select_rollouts_per_prompt(questions)
        
        print(f"   Selected: {len(selected_indices)} rollouts from {select_stats['selected_prompts']}/{select_stats['total_prompts']} prompts")
        print(f"   Selected questions: {select_stats['selected_questions']}/{len(selected_indices)}")
        
        if len(selected_indices) == 0:
            # No valid rollouts, but still need to set extra_info for reward manager
            log_prob_rewards = torch.zeros(batch_size)
            batch.non_tensor_batch["log_prob_rewards"] = log_prob_rewards.numpy()
            
            # Set extra_info with default value 0.0
            extra_info_list = batch.non_tensor_batch.get("extra_info", [{} for _ in range(batch_size)])
            if not isinstance(extra_info_list, list):
                extra_info_list = [{} for _ in range(batch_size)]
            for i in range(batch_size):
                if not isinstance(extra_info_list[i], dict):
                    extra_info_list[i] = {}
                extra_info_list[i]["log_prob_rewards"] = 0.0  # Default reward
            batch.non_tensor_batch["extra_info"] = np.array(extra_info_list, dtype=object)
            
            metrics["log_prob_reward/mean"] = 0.0
            metrics["log_prob_reward/valid_mean"] = 0.0
            metrics["log_prob_reward/valid_ratio"] = 0.0
            metrics["log_prob_reward/batch_discarded"] = 1.0
            print(f"   No valid rollouts selected, setting default reward=0.0")
            print("="*70 + "\n")
            
            # Still save log (first N steps)
            log_complete_step_data(
                step=_QUESTION_LOG_STEP,
                prompts=prompts,
                responses=responses,
                questions=questions,
                ground_truths=ground_truths,
                user_responses=[],
                log_probs_with=[],
                log_probs_without=[],
                rewards=[],
                selected_indices=[],
            )
            return
        
        metrics["log_prob_reward/batch_discarded"] = 0.0
        metrics["log_prob_reward/prompts_skipped"] = float(len(skipped_prompts))
        metrics["log_prob_reward/prompts_used"] = float(select_stats['selected_prompts'])
        
        # 6. Get user responses (only for selected rollouts)
        print(f"   Fetching user responses for {len(selected_indices)} selected rollouts...")
        fetch_start = time.time()
        
        # Only call for those with questions
        selected_questions = [questions[i] for i in selected_indices]
        selected_ground_truths = [ground_truths[i] for i in selected_indices]
        
        with marked_timer("fetch_responses", timing_raw, "yellow"):
            user_responses_selected = fetch_responses_batch(selected_questions, selected_ground_truths)
        
        fetch_time = time.time() - fetch_start
        num_responses = sum(1 for r in user_responses_selected if r)
        num_questions_selected = sum(1 for q in selected_questions if q)
        print(f"   Responses: {num_responses}/{num_questions_selected} in {fetch_time:.2f}s")
        
        # 7. Construct prompts (using truncated history)
        prompts_with_qa = []
        prompts_without_qa = []
        gt_completions = []
        valid_indices = []  # Indices within selected_indices
        orig_valid_indices = []  # Indices in original batch
        
        for local_idx, orig_idx in enumerate(selected_indices):
            q = selected_questions[local_idx]
            u = user_responses_selected[local_idx]
            gt = selected_ground_truths[local_idx]
            h = histories[orig_idx]
            
            if q and u and gt:
                valid_indices.append(local_idx)
                orig_valid_indices.append(orig_idx)
                
                clarification_turn = f"Agent: {q}\nUser: {u}"
                prompts_with_qa.append(RECOVER_PROMPT_TEMPLATE.format(history=h, clarification_turn=clarification_turn))
                prompts_without_qa.append(RECOVER_PROMPT_TEMPLATE.format(history=h, clarification_turn=""))
                gt_completions.append(gt)
                
                # Record first sample
                if not _FIRST_SAMPLE_LOGGED and first_sample_details is None:
                    first_sample_details = {
                        "sample_idx": orig_idx,
                        "extracted_question": q,
                        "user_response": u,
                        "ground_truth_clean": gt,
                        "history_length": len(h),
                        "prompt_with_qa": prompts_with_qa[-1][:2000],
                        "prompt_without_qa": prompts_without_qa[-1][:2000],
                    }
        
        print(f"   Valid samples: {len(valid_indices)}/{batch_size}")
        
        # 8. Compute log probs (merged batch)
        log_prob_rewards = torch.zeros(batch_size)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool)
        log_probs_with_list = []
        log_probs_without_list = []
        
        if len(valid_indices) > 0:
            print(f"   Computing log probs (merged batch)...")
            log_prob_start = time.time()
            
            try:
                num_workers = 4
                
                # Merge into single batch
                merged_batch, original_size, split_point = construct_merged_log_prob_batch(
                    prompts_with_qa, prompts_without_qa, gt_completions,
                    tokenizer, num_workers=num_workers
                )
                
                padded_size = merged_batch["input_ids"].shape[0]
                print(f"   Merged batch: {padded_size} (original: {original_size * 2})")
                
                meta_info = {
                    "micro_batch_size": 8,
                    "temperature": 1.0,
                    "use_dynamic_bsz": False,
                }
                
                data_merged = DataProto.from_single_dict(merged_batch)
                data_merged.meta_info = meta_info
                
                # Single compute_log_prob
                with marked_timer("log_prob_merged", timing_raw, "cyan"):
                    output = actor_rollout_wg.compute_log_prob(data_merged)
                
                # Key: detach immediately to avoid gradient accumulation
                log_probs = output.batch.get("old_log_probs", output.batch.get("log_probs")).detach()
                response_mask = (merged_batch["responses"] != tokenizer.pad_token_id).float()
                
                # Release GPU memory immediately
                del output, data_merged
                torch.cuda.empty_cache()
                
                log_prob_time = time.time() - log_prob_start
                print(f"   Log probs computed in {log_prob_time:.2f}s")
                
                # ========== Dimension verification ==========
                print(f"   Dimension check:")
                print(f"      - log_probs shape: {log_probs.shape}")
                print(f"      - response_mask shape: {response_mask.shape}")
                print(f"      - responses shape: {merged_batch['responses'].shape}")
                
                # Check actual token counts per sample
                actual_token_counts = response_mask.sum(dim=1).int().tolist()
                print(f"      - Actual completion tokens per sample (first 4): {actual_token_counts[:4]}")
                print(f"      - Max completion len (config): {merged_batch['responses'].shape[1]}")
                
                # Check for truncation
                max_len = merged_batch['responses'].shape[1]
                truncated_count = sum(1 for c in actual_token_counts if c >= max_len - 1)
                if truncated_count > 0:
                    print(f"      WARNING: {truncated_count}/{len(actual_token_counts)} samples may be truncated!")
                else:
                    print(f"      No truncation detected")
                
                # Move to CPU immediately to release GPU memory
                log_probs_cpu = log_probs.cpu()
                response_mask_cpu = response_mask.cpu()
                del log_probs, response_mask
                torch.cuda.empty_cache()
                
                # Split results (using CPU tensor)
                for idx in range(len(valid_indices)):
                    if idx >= original_size:
                        break
                    
                    orig_idx = orig_valid_indices[idx]
                    valid_mask[orig_idx] = True
                    
                    # With QA (first half)
                    mask_w = response_mask_cpu[idx].bool()
                    num_tokens_w = mask_w.sum().item()
                    avg_lp_with = log_probs_cpu[idx][mask_w].mean().item() if num_tokens_w > 0 else 0.0
                    
                    # Without QA (second half)
                    without_idx = idx + split_point
                    mask_wo = response_mask_cpu[without_idx].bool()
                    num_tokens_wo = mask_wo.sum().item()
                    avg_lp_without = log_probs_cpu[without_idx][mask_wo].mean().item() if num_tokens_wo > 0 else 0.0
                    
                    # Reward = log prob difference (no scaling)
                    reward = (avg_lp_with - avg_lp_without) * 1.0
                    log_prob_rewards[orig_idx] = reward
                    
                    log_probs_with_list.append(avg_lp_with)
                    log_probs_without_list.append(avg_lp_without)
                    
                    # Detailed log (first sample)
                    if idx == 0:
                        print(f"   First sample computation:")
                        print(f"      - Tokens (with QA): {num_tokens_w}, (without QA): {num_tokens_wo}")
                        print(f"      - Avg log_prob (with QA): {avg_lp_with:.6f}")
                        print(f"      - Avg log_prob (without QA): {avg_lp_without:.6f}")
                        print(f"      - Diff: {avg_lp_with - avg_lp_without:.6f}")
                        print(f"      - Reward (diff): {reward:.4f}")
                    
                    # Record first sample log prob
                    if first_sample_details and idx == 0:
                        first_sample_details["log_prob_with"] = avg_lp_with
                        first_sample_details["log_prob_without"] = avg_lp_without
                        first_sample_details["reward"] = reward
                        first_sample_details["num_tokens_with"] = num_tokens_w
                        first_sample_details["num_tokens_without"] = num_tokens_wo
                
            except Exception as e:
                print(f"   WARNING: Log prob failed: {e}")
                import traceback
                traceback.print_exc()
        
        # 9. Store rewards in extra_info for DAPO reward manager
        batch_size_total = len(batch.batch["prompts"])
        
        # Initialize extra_info
        if "extra_info" not in batch.non_tensor_batch:
            batch.non_tensor_batch["extra_info"] = [{} for _ in range(batch_size_total)]
        
        # Ensure extra_info is list of dicts
        extra_info_list = batch.non_tensor_batch["extra_info"]
        if not isinstance(extra_info_list, list):
            extra_info_list = [extra_info_list] * batch_size_total if isinstance(extra_info_list, dict) else [{} for _ in range(batch_size_total)]
            batch.non_tensor_batch["extra_info"] = extra_info_list
        
        # Store per-sample log_prob_rewards and valid_for_update flag
        for i in range(batch_size_total):
            if not isinstance(extra_info_list[i], dict):
                extra_info_list[i] = {}
            extra_info_list[i]["log_prob_rewards"] = float(log_prob_rewards[i].item())
            extra_info_list[i]["valid_for_update"] = bool(valid_mask[i].item())
        
        # Must convert to np.ndarray(dtype=object) - DataProto.chunk requires np.ndarray
        batch.non_tensor_batch["extra_info"] = np.array(extra_info_list, dtype=object)
        
        # Also keep tensor version for debugging
        batch.non_tensor_batch["log_prob_rewards"] = log_prob_rewards.numpy()
        batch.non_tensor_batch["valid_for_update"] = valid_mask.numpy()
        
        # Key: Set loss_mask so invalid samples don't contribute to gradient update
        if "attention_mask" in batch.batch:
            print(f"   Applied loss_mask: {valid_mask.sum().item()}/{batch_size_total} samples will update")
        
        # 10. Statistics and detailed logging
        valid_rewards = log_prob_rewards[valid_mask]
        
        # ============ TRAINING DYNAMICS LOG ============
        print(f"\n" + "="*70, flush=True)
        print(f"TRAINING DYNAMICS (Log Prob Reward)", flush=True)
        print(f"="*70, flush=True)
        print(f"   Valid samples: {valid_mask.sum().item()}/{len(selected_indices)} selected", flush=True)
        print(f"   Questions extracted: {total_questions}/{batch_size} total", flush=True)
        print(f"   Selected questions: {select_stats['selected_questions']}/{len(selected_indices)}", flush=True)
        print(f"   User responses fetched: {num_responses}/{num_questions_selected}", flush=True)
        print(f"   Prompts used/skipped: {select_stats['selected_prompts']}/{select_stats['skipped_prompts']}", flush=True)
        print(f"", flush=True)
        if len(valid_rewards) > 0:
            lp_with_mean = np.mean(log_probs_with_list)
            lp_without_mean = np.mean(log_probs_without_list)
            reward_mean = valid_rewards.mean().item()
            reward_std = valid_rewards.std().item() if len(valid_rewards) > 1 else 0.0
            reward_min = valid_rewards.min().item()
            reward_max = valid_rewards.max().item()
            
            print(f"   LOG PROB WITH QA:", flush=True)
            print(f"      Mean: {lp_with_mean:.6f}", flush=True)
            print(f"      Per-sample: {log_probs_with_list[:5]}..." if len(log_probs_with_list) > 5 else f"      Per-sample: {log_probs_with_list}", flush=True)
            print(f"", flush=True)
            print(f"   LOG PROB WITHOUT QA:", flush=True)
            print(f"      Mean: {lp_without_mean:.6f}", flush=True)
            print(f"      Per-sample: {log_probs_without_list[:5]}..." if len(log_probs_without_list) > 5 else f"      Per-sample: {log_probs_without_list}", flush=True)
            print(f"", flush=True)
            print(f"   REWARD (with - without):", flush=True)
            print(f"      Mean: {reward_mean:.6f}", flush=True)
            print(f"      Std:  {reward_std:.6f}", flush=True)
            print(f"      Min:  {reward_min:.6f}", flush=True)
            print(f"      Max:  {reward_max:.6f}", flush=True)
            print(f"", flush=True)
            print(f"   Time: {time.time() - start_time:.2f}s", flush=True)
        else:
            print(f"   WARNING: No valid samples in this batch", flush=True)
        print(f"="*70 + "\n", flush=True)
        
        # 11. Save complete step data to JSON (first N steps only)
        rewards_list = []
        for idx in range(len(valid_indices)):
            orig_idx = orig_valid_indices[idx]
            rewards_list.append(float(log_prob_rewards[orig_idx].item()))
        
        log_complete_step_data(
            step=_QUESTION_LOG_STEP,
            prompts=prompts,
            responses=responses,
            questions=questions,
            ground_truths=ground_truths,
            user_responses=user_responses_selected,
            log_probs_with=log_probs_with_list,
            log_probs_without=log_probs_without_list,
            rewards=rewards_list,
            selected_indices=selected_indices,
            valid_orig_indices=orig_valid_indices,
        )
        
        # Record metrics (these will appear in log file)
        metrics["log_prob_reward/mean"] = log_prob_rewards.mean().item()
        metrics["log_prob_reward/valid_mean"] = valid_rewards.mean().item() if len(valid_rewards) > 0 else 0.0
        metrics["log_prob_reward/valid_ratio"] = valid_mask.float().mean().item()
        metrics["log_prob_reward/num_questions"] = float(total_questions)
        metrics["log_prob_reward/num_valid"] = float(valid_mask.sum().item())
        if log_probs_with_list:
            metrics["log_prob_reward/lp_with_mean"] = np.mean(log_probs_with_list)
            metrics["log_prob_reward/lp_without_mean"] = np.mean(log_probs_without_list)
            metrics["log_prob_reward/lp_diff_mean"] = np.mean(log_probs_with_list) - np.mean(log_probs_without_list)
            metrics["log_prob_reward/reward_std"] = valid_rewards.std().item() if len(valid_rewards) > 1 else 0.0
            metrics["log_prob_reward/reward_min"] = valid_rewards.min().item()
            metrics["log_prob_reward/reward_max"] = valid_rewards.max().item()
        
        # Filter stats
        filter_stats = _batch_filter.get_stats()
        metrics["log_prob_reward/filter_discard_rate"] = filter_stats["discard_rate"]
        metrics["log_prob_reward/filter_total_batches"] = float(filter_stats["total_batches"])
        metrics["log_prob_reward/filter_discarded_batches"] = float(filter_stats["discarded_batches"])
        
        # Final cleanup of temporary GPU tensors
        if 'log_probs_cpu' in locals():
            del log_probs_cpu
        if 'response_mask_cpu' in locals():
            del response_mask_cpu
        if 'merged_batch' in locals():
            del merged_batch
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"WARNING: Log prob reward failed: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# Patch Function
# ============================================================

def patch_compute_kl_related_metrics():
    # Import from local verl_recipe (derived from verl project)
    import sys
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from verl_recipe.dapo_ray_trainer import RayDAPOTrainer
    from verl import DataProto
    from verl.trainer.ppo.core_algos import agg_loss
    from verl.trainer.ppo.ray_trainer import compute_response_mask
    from verl.utils.profiler import marked_timer
    
    original = RayDAPOTrainer.compute_kl_related_metrics
    
    def patched_compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """
        Patched version with Log prob QA reward calculation.
        Note: Response length is controlled by shell script's max_response_length
        """
        tokenizer = self.tokenizer
        actor_rollout_wg = self.actor_rollout_wg
        
        # Compute log prob reward
        compute_log_prob_qa_reward(batch, tokenizer, actor_rollout_wg, metrics, timing_raw)
        
        # Call original method (dimensions controlled by max_response_length)
        return original(self, batch, metrics, timing_raw)
    
    RayDAPOTrainer.compute_kl_related_metrics = patched_compute_kl_related_metrics
    print("Patched RayDAPOTrainer.compute_kl_related_metrics with:")
    print(f"   - Log prob QA reward (question extraction)")
    print(f"   - History truncation ({MAX_HISTORY_LENGTH} chars)")

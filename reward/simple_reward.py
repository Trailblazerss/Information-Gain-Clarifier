#!/usr/bin/env python3


import numpy as np


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None):
    """Return the per-sample reward for DAPO.

    In IG-Clarifier-style training, the actual log-prob PMI reward is computed
    inside the patched trainer loop and attached to each sample's `extra_info`
    under key `log_prob_rewards`. The custom reward function simply surfaces
    that value to verl's reward manager.
    """
    if extra_info is None:
        return 0.0
    if isinstance(extra_info, dict):
        try:
            return float(extra_info.get("log_prob_rewards", 0.0) or 0.0)
        except Exception:
            return 0.0
    return 0.0


def compute_score_batch(data_sources, solution_strs, ground_truths=None, extra_infos=None):

    if ground_truths is None:
        ground_truths = [None] * len(data_sources)
    if extra_infos is None:
        extra_infos = [None] * len(data_sources)
    
    scores = []
    for ds, sol, gt, ei in zip(data_sources, solution_strs, ground_truths, extra_infos):
        scores.append(compute_score(ds, sol, gt, ei))
    
    return scores

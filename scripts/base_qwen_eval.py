#!/usr/bin/env python3
"""Paper-faithful τ-Bench baseline runner for Qwen3-8B / None.

This runner keeps the baseline close to the paper and upstream τ-Bench:
- raw tool-calling agent
- user simulator at temperature 1.0
- agent temperature 0.01
- strips persisted <think> traces from the replayed history to keep long
  conversations within the model context budget
- explicit per-seed sampling control for both agent and user completions

The matching vLLM server should be started with reasoning disabled by default:
```
--default-chat-template-kwargs '{"enable_thinking": false}'
```
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(content: str) -> str:
    if "<think" not in content:
        return content
    stripped = _THINK_BLOCK_RE.sub("", content)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def _sanitize_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    sanitized: list[Any] = []
    for message in messages:
        if isinstance(message, dict):
            msg = dict(message)
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = _strip_thinking(content)
            sanitized.append(msg)
        else:
            sanitized.append(message)
    return sanitized


def _patch_completion(module: Any, *, temperature: float, seed: int | None) -> None:
    original = getattr(module, "_baseline_original_completion", module.completion)
    module._baseline_original_completion = original

    def configured_completion(*args: Any, **kwargs: Any) -> Any:
        if "messages" in kwargs:
            kwargs["messages"] = _sanitize_messages(kwargs["messages"])
        kwargs.setdefault("temperature", temperature)
        if seed is not None:
            kwargs.setdefault("seed", seed)
        extra_body = dict(kwargs.get("extra_body") or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs["enable_thinking"] = True
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        kwargs["extra_body"] = extra_body
        return original(*args, **kwargs)

    module.completion = configured_completion


def _patch_tau_bench_temperatures(*, agent_temperature: float, user_temperature: float, seed: int | None) -> None:
    import tau_bench.agents.tool_calling_agent as tool_calling_agent
    import tau_bench.envs.user as user_env

    _patch_completion(tool_calling_agent, temperature=agent_temperature, seed=seed)
    _patch_completion(user_env, temperature=user_temperature, seed=seed)


def _pass_rate(rows: list[Any]) -> float:
    if not rows:
        return 0.0
    success = sum(1 for row in rows if abs(float(getattr(row, "reward", 0.0)) - 1.0) <= 1e-6)
    return success / len(rows)


def _avg_reward(rows: list[Any]) -> float:
    if not rows:
        return 0.0
    return sum(float(getattr(row, "reward", 0.0)) for row in rows) / len(rows)


def _run_one(
    *,
    env: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from tau_bench.run import run as tau_run
    from tau_bench.types import RunConfig

    _patch_tau_bench_temperatures(
        agent_temperature=args.agent_temperature,
        user_temperature=args.user_temperature,
        seed=seed,
    )

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["OPENAI_BASE_URL"] = args.base_url
    os.environ["OPENAI_API_BASE"] = args.base_url

    seed_log_dir = Path(args.log_root) / f"base_qwen_{env}_seed{seed}"
    seed_log_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        model_provider="openai",
        user_model_provider="openai",
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=env,
        agent_strategy="tool-calling",
        temperature=args.agent_temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=str(seed_log_dir),
        max_concurrency=args.max_concurrency,
        seed=seed,
        shuffle=args.shuffle,
        user_strategy="llm",
        few_shot_displays_path=None,
    )

    print()
    print("=" * 72)
    print(f"Running base_qwen baseline: env={env} seed={seed}")
    print(f"Base URL: {args.base_url}")
    print(f"Result dir: {seed_log_dir}")
    print("=" * 72)
    rows = tau_run(config)
    summary = {
        "env": env,
        "seed": seed,
        "num_rows": len(rows),
        "pass_at_1": _pass_rate(rows),
        "avg_reward": _avg_reward(rows),
        "result_dir": str(seed_log_dir),
    }
    with open(seed_log_dir / "seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Seed summary: {json.dumps(summary, sort_keys=True)}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-faithful τ-Bench base_qwen baseline runner.")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["retail", "airline"],
        choices=["retail", "airline"],
        help="Environments to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Seed values for repeated runs.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:18000/v1"),
        help="OpenAI-compatible base URL for the Qwen3-8B server.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Agent model name exposed by the server.",
    )
    parser.add_argument(
        "--user-model",
        default="Qwen/Qwen3-8B",
        help="User-simulator model name exposed by the server.",
    )
    parser.add_argument(
        "--agent-temperature",
        type=float,
        default=0.01,
        help="Agent sampling temperature. Paper baseline uses 0.01.",
    )
    parser.add_argument(
        "--user-temperature",
        type=float,
        default=1.0,
        help="User-simulator sampling temperature. Paper baseline uses 1.0.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Number of tasks to run in parallel.",
    )
    parser.add_argument(
        "--task-split",
        default="test",
        choices=["train", "test", "dev"],
        help="τ-Bench task split.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="τ-Bench trials per seed.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First task index to evaluate.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="One-past-last task index to evaluate (-1 means all tasks).",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        help="Optional explicit task ids.",
    )
    parser.add_argument(
        "--shuffle",
        type=int,
        default=0,
        help="Shuffle task order within a seed (0 or 1).",
    )
    parser.add_argument(
        "--log-root",
        default="results",
        help="Root directory for τ-Bench result files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    Path(args.log_root).mkdir(parents=True, exist_ok=True)

    overall: dict[str, Any] = {
        "base_url": args.base_url,
        "agent_temperature": args.agent_temperature,
        "user_temperature": args.user_temperature,
        "seeds": args.seeds,
        "envs": {},
    }

    for env in args.envs:
        env_summaries: list[dict[str, Any]] = []
        for seed in args.seeds:
            env_summaries.append(_run_one(env=env, seed=seed, args=args))
        pass_rates = [summary["pass_at_1"] for summary in env_summaries]
        rewards = [summary["avg_reward"] for summary in env_summaries]
        overall["envs"][env] = {
            "runs": env_summaries,
            "mean_pass_at_1": mean(pass_rates) if pass_rates else 0.0,
            "stdev_pass_at_1": pstdev(pass_rates) if len(pass_rates) > 1 else 0.0,
            "mean_avg_reward": mean(rewards) if rewards else 0.0,
        }

        print()
        print(
            f"[summary] {env}: mean pass@1={overall['envs'][env]['mean_pass_at_1']:.4f} "
            f"(stdev={overall['envs'][env]['stdev_pass_at_1']:.4f})"
        )

    summary_path = Path(args.log_root) / "base_qwen_paper_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, sort_keys=True)
    print()
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()

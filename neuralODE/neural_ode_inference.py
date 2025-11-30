import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from neuralODE.neural_ode_density import (
    ContinuousNormalizingFlow,
    NPZTargetDataset,
    ODEFunc,
)


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


@dataclass
class EvalConfig:
    npz_path: str
    model_path: str
    batch_size: int = 512
    max_samples: int = 0  # 0 means evaluate all samples
    hidden_dims: Tuple[int, ...] = (512, 512)
    activation: str = "silu"
    time_dependent: bool = True
    solver: str = "dopri5"
    t0: float = 0.0
    t1: float = 1.0
    rtol: float = 1e-5
    atol: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_metrics: str = ""
    save_logp: str = ""


def load_flow(cfg: EvalConfig, target_dim: int) -> ContinuousNormalizingFlow:
    odefunc = ODEFunc(
        dim=target_dim,
        hidden_dims=cfg.hidden_dims,
        activation=cfg.activation,
        time_dependent=cfg.time_dependent,
    ).to(cfg.device)

    flow = ContinuousNormalizingFlow(
        func=odefunc,
        t0=cfg.t0,
        t1=cfg.t1,
        solver=cfg.solver,
        rtol=cfg.rtol,
        atol=cfg.atol,
    ).to(cfg.device)

    chkpt = torch.load(cfg.model_path, map_location=cfg.device)
    state_dict = chkpt.get("model_state_dict") if isinstance(chkpt, dict) else None
    if state_dict is None:
        state_dict = chkpt
    flow.load_state_dict(state_dict)
    flow.eval()
    return flow


def evaluate(cfg: EvalConfig) -> None:
    full_dataset = NPZTargetDataset(cfg.npz_path)
    target_dim = full_dataset.target_dim
    if cfg.max_samples > 0:
        num_samples = min(cfg.max_samples, len(full_dataset))
        dataset = Subset(full_dataset, list(range(num_samples)))
        print(f"[Eval] Limiting evaluation to first {num_samples} samples")
    else:
        dataset = full_dataset
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    flow = load_flow(cfg, target_dim)

    total_logp = 0.0
    total_samples = 0
    all_logp = []

    torch.set_grad_enabled(True)
    for batch in loader:
        x = batch.to(cfg.device)
        logp = flow.log_prob(x)
        total_logp += logp.sum().item()
        total_samples += x.size(0)
        if cfg.save_logp:
            all_logp.append(logp.detach().cpu())

    avg_logp = total_logp / max(total_samples, 1)
    nll = -avg_logp

    print(
        f"[Eval] samples {total_samples}  avg_logp {avg_logp:.6f}  NLL {nll:.6f}"
    )

    metrics = {
        "num_samples": int(total_samples),
        "avg_logp": float(avg_logp),
        "nll": float(nll),
    }

    if cfg.save_metrics:
        ensure_parent_dir(cfg.save_metrics)
        with open(cfg.save_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Eval] wrote metrics to {cfg.save_metrics}")

    if cfg.save_logp:
        ensure_parent_dir(cfg.save_logp)
        stacked = torch.cat(all_logp).numpy() if all_logp else np.zeros(0, dtype=np.float32)
        np.save(cfg.save_logp, stacked)
        print(f"[Eval] wrote per-sample logp to {cfg.save_logp}")


def parse_args() -> EvalConfig:
    config_only = argparse.ArgumentParser(add_help=False)
    config_only.add_argument("--config", type=str, default="")
    known, _ = config_only.parse_known_args()

    yaml_defaults = {}
    if getattr(known, "config", "") and yaml is not None:
        try:
            with open(known.config, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)
            if isinstance(y, dict):
                yaml_defaults = y
        except Exception as e:
            print(f"Failed to load YAML config: {e}")

    def dget(key, default):
        return yaml_defaults.get(key, default)

    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNF and report dataset NLL",
        parents=[config_only],
    )
    parser.add_argument("--npz", required=("npz" not in yaml_defaults), default=dget("npz", None))
    parser.add_argument("--model", required=("model" not in yaml_defaults), default=dget("model", None))
    parser.add_argument("--batch", type=int, default=dget("batch", 512))
    parser.add_argument("--max-samples", type=int, default=dget("max_samples", 0), help="Limit evaluation to first N samples (0 = all samples)")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=dget("hidden_dims", [512, 512]))
    parser.add_argument("--activation", type=str, default=dget("activation", "silu"), choices=["silu", "tanh"])
    parser.add_argument("--time-dependent", dest="time_dependent", action="store_true")
    parser.add_argument("--no-time-dependent", dest="time_dependent", action="store_false")
    parser.add_argument("--solver", type=str, default=dget("solver", "dopri5"))
    parser.add_argument("--t0", type=float, default=dget("t0", 0.0))
    parser.add_argument("--t1", type=float, default=dget("t1", 1.0))
    parser.add_argument("--rtol", type=float, default=dget("rtol", 1e-5))
    parser.add_argument("--atol", type=float, default=dget("atol", 1e-5))
    parser.add_argument("--device", type=str, default=dget("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--save-metrics", type=str, default=dget("save_metrics", ""))
    parser.add_argument("--save-logp", type=str, default=dget("save_logp", ""))

    parser.set_defaults(time_dependent=dget("time_dependent", True))

    args = parser.parse_args()
    hidden_dims = tuple(args.hidden_dims) if isinstance(args.hidden_dims, list) else tuple([args.hidden_dims])

    return EvalConfig(
        npz_path=args.npz,
        model_path=args.model,
        batch_size=args.batch,
        max_samples=args.max_samples,
        hidden_dims=hidden_dims,
        activation=args.activation,
        time_dependent=args.time_dependent,
        solver=args.solver,
        t0=args.t0,
        t1=args.t1,
        rtol=args.rtol,
        atol=args.atol,
        device=args.device,
        save_metrics=args.save_metrics,
        save_logp=args.save_logp,
    )


    return flow, cfg


if __name__ == "__main__":
    config = parse_args()
    evaluate(config)


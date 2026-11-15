"""
Slice-optimiser rApp — decision core.

Single-step PPO policy, extracted from the validated RANOpt v17.8 design
(RANOpt_AI_SIM_v17.8_fixed.html):

  * State  : the seven network KPIs the operator already collects as PM
             counters, normalised by their targets, plus cell load.
  * Action : one of a discrete set of scheduling profiles (PRB-share
             vectors across eMBB / URLLC / mMTC slices + scheduler mode).
  * Reward : the validated weighted formula from the fixed code —
             reward = 100 * sum(w_i * min(1, kpi_i / target_i)), capped
             at 100, with weights [0.22, 0.13, 0.17, 0.10, 0.13, 0.10,
             0.15] over [median DL, median UL, edge DL, edge UL,
             fairness, PRB utilisation (inverted), signalling (inverted)].

"Single-step" means each decision is scored on its immediate outcome —
the contextual-bandit regime the CCNC 2027 paper analyses under
"guarded acceptance": a policy action is only applied if it clears the
guardrail floors, otherwise the rule-based profile is used instead.
"""

import numpy as np
import torch
import torch.nn as nn

# --- Validated constants (RANOpt_AI_SIM_v17.8_fixed.html) -------------------
KPI_NAMES = ["median_dl", "median_ul", "edge_dl", "edge_ul",
             "fairness", "prb_util", "signalling"]
DEFAULT_TARGETS = np.array([250.0, 75.0, 75.0, 25.0, 0.92, 88.0, 30.0])
REWARD_WEIGHTS = np.array([0.22, 0.13, 0.17, 0.10, 0.13, 0.10, 0.15])
# prb_util and signalling are "lower is better": ratio uses target/value.
INVERTED = np.array([False, False, False, False, False, True, True])

# Scheduling profiles: (name, [eMBB, URLLC, mMTC] PRB shares, scheduler mode)
PROFILES = [
    ("balanced",       [0.60, 0.20, 0.20], "proportional_fair"),
    ("dl_heavy",       [0.75, 0.15, 0.10], "proportional_fair"),
    ("edge_protect",   [0.50, 0.25, 0.25], "round_robin_edge_boost"),
    ("fairness_first", [0.45, 0.25, 0.30], "fair_queueing"),
    ("urllc_priority", [0.40, 0.45, 0.15], "strict_priority"),
    ("energy_lean",    [0.55, 0.20, 0.25], "proportional_fair_sleep"),
]
N_ACTIONS = len(PROFILES)
STATE_DIM = len(KPI_NAMES) + 1  # KPI ratios + cell load


def kpi_ratios(kpis: np.ndarray, targets: np.ndarray = DEFAULT_TARGETS) -> np.ndarray:
    """Per-KPI achievement ratios, capped at 1 (as in the validated code)."""
    ratios = np.where(INVERTED, targets / np.maximum(kpis, 1e-6), kpis / targets)
    return np.minimum(ratios, 1.0)


def reward(kpis: np.ndarray, targets: np.ndarray = DEFAULT_TARGETS) -> float:
    """The validated reward: 100 * weighted sum of capped ratios."""
    return float(min(100.0, 100.0 * np.dot(REWARD_WEIGHTS, kpi_ratios(kpis, targets))))


def make_state(kpis: np.ndarray, load: float,
               targets: np.ndarray = DEFAULT_TARGETS) -> np.ndarray:
    return np.concatenate([kpi_ratios(kpis, targets), [load]]).astype(np.float32)


class PolicyNet(nn.Module):
    """Small MLP policy — inference is a sub-millisecond forward pass."""

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class SingleStepPPO:
    """PPO for the single-step (contextual bandit) regime: one decision,
    immediate reward, clipped-surrogate update — as in RANOpt v17.8."""

    def __init__(self, lr: float = 3e-3, clip: float = 0.2, seed: int = 0):
        torch.manual_seed(seed)
        self.policy = PolicyNet()
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.clip = clip

    def act(self, state: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            logits = self.policy(torch.tensor(state).unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        if deterministic:
            return int(torch.argmax(probs)), probs.numpy()
        return int(torch.multinomial(probs, 1)), probs.numpy()

    def update(self, states, actions, rewards_, epochs: int = 4):
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        adv = torch.tensor(rewards_, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        with torch.no_grad():
            old_logp = torch.log_softmax(self.policy(states), dim=-1) \
                .gather(1, actions.unsqueeze(1)).squeeze(1)
        for _ in range(epochs):
            logp = torch.log_softmax(self.policy(states), dim=-1) \
                .gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(logp - old_logp)
            loss = -torch.min(ratio * adv,
                              torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path, weights_only=True))
        self.policy.eval()

"""Bayesian engine for selecting and updating response strategies.

Implements a lightweight contextual bandit (Thompson sampling style) over
presentation strategies. Supports:
- soft preferences (bias, not lock)
- optional hard lock (user can disable exploration)
- Option B corrective exploration after negative feedback:
    * temperature boost
    * strong repeat penalty
    * posterior damping (reduce confidence in last chosen arm)
"""

import numpy as np
import random

from . import config
from .utils import sigmoid, mean_uncertainty


class BayesianEngine:
    def __init__(self):
        # Keep posterior state for all known strategies (enabled + disabled).
        # Selection/visualization should still use `config.STRATEGY_NAMES` (enabled only).
        initial_ids = set(getattr(config, "STRATEGY_ITEMS", {}) or {}).union(set(config.STRATEGY_NAMES))
        self.global_mu = {k: np.zeros(config.D) for k in initial_ids}
        self.global_sinv = {k: np.eye(config.D) * 0.1 for k in initial_ids}
        self.users = {}
        self.global_n = 0

    def _all_strategy_ids(self) -> set[str]:
        items = getattr(config, "STRATEGY_ITEMS", {}) or {}
        return set(items.keys()) if items else set(config.STRATEGY_NAMES)

    def reconcile_strategies(self) -> None:
        """
        Reconcile engine posterior dictionaries with the current strategy store.

        - If new strategies appear: add priors (zero mean, low precision identity).
        - If strategies are hard-deleted: remove their posterior state.
        - If strategies are disabled: keep state but they won't be selected/visualized
          because selection/summary loops use `config.STRATEGY_NAMES`.
        """
        desired_ids = self._all_strategy_ids()

        # Add missing strategy keys.
        for sid in desired_ids:
            if sid not in self.global_mu:
                self.global_mu[sid] = np.zeros(config.D)
                self.global_sinv[sid] = np.eye(config.D) * 0.1

        # Remove hard-deleted strategies.
        current_ids = set(self.global_mu.keys())
        to_remove = current_ids - desired_ids
        for sid in to_remove:
            self.global_mu.pop(sid, None)
            self.global_sinv.pop(sid, None)
            for user in self.users.values():
                user.get("mu", {}).pop(sid, None)
                user.get("sigma_inv", {}).pop(sid, None)
                prefs = user.get("prefs")
                if isinstance(prefs, set) and sid in prefs:
                    prefs.discard(sid)
                if user.get("locked_strategy") == sid:
                    user["locked_strategy"] = None
                if user.get("pending_strategy") == sid:
                    user["pending_strategy"] = None

        # Ensure all existing users have keys for every known strategy.
        for user in self.users.values():
            mu = user.get("mu") or {}
            sinv = user.get("sigma_inv") or {}
            for sid in desired_ids:
                if sid not in mu:
                    mu[sid] = self.global_mu[sid].copy()
                if sid not in sinv:
                    sinv[sid] = self.global_sinv[sid].copy()
            user["mu"] = mu
            user["sigma_inv"] = sinv

    def _new_user(self):
        all_ids = self._all_strategy_ids()
        return {
            "mu":            {k: self.global_mu[k].copy()   for k in all_ids},
            "sigma_inv":     {k: self.global_sinv[k].copy() for k in all_ids},
            "history":       [],
            "reward_log":    [],
            "last_message":  "",
            "last_response": "",
            "last_strategy": None,
            "last_x":        None,
            "msg_count":     0,
            "prefs":         set(),
            "locked_strategy": None,
            "pending_strategy": None,
        }

    def get_user(self, uid: str):
        if uid not in self.users:
            self.users[uid] = self._new_user()
        return self.users[uid]

    def featurize(self, message: str, user: dict) -> np.ndarray:
        words   = message.split()
        msg_len = min(len(message) / 500, 1.0)
        word_ct = min(len(words) / 100, 1.0)
        has_q   = 1.0 if "?" in message else 0.0
        is_long = 1.0 if len(words) > 40 else 0.0
        informal = {"lol","gonna","wanna","yo","omg","idk","wtf","lmao"}
        formal  = 0.0 if any(w in message.lower() for w in informal) else 1.0
        rl      = user["reward_log"]
        avg_r   = sum(r for _, r in rl[-5:]) / max(len(rl[-5:]), 1) if rl else 0.5
        msg_num = min(user["msg_count"] / 20.0, 1.0)
        # Map last strategy into [0,1] by its rank among enabled strategies.
        # If the strategy is currently disabled, fall back to 0.5.
        if user.get("last_strategy") and user["last_strategy"] in config.STRATEGY_NAMES and config.K > 1:
            si = config.STRATEGY_NAMES.index(user["last_strategy"]) / (config.K - 1)
        else:
            si = 0.5
        trend   = 0.0
        if len(rl) >= 3:
            ys = [r for _, r in rl[-5:]]
            xs = list(range(len(ys)))
            mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
            num = sum((xi-mx)*(yi-my) for xi, yi in zip(xs, ys))
            den = sum((xi-mx)**2 for xi in xs) or 1e-8
            trend = float(np.clip(num/den, -1, 1))
        # last dim is just a small noise to break ties
        return np.array([msg_len, word_ct, has_q, is_long,
                         formal, avg_r, msg_num, si, trend, random.random()])

    def _damp_posterior(self, user: dict, strategy: str, strength: float):
        """Option B: reduce confidence in the last chosen strategy."""
        if strategy not in user.get("mu", {}):
            return
        s = float(np.clip(strength, 0.0, 1.0))
        if s <= 0:
            return
        # shrink mean magnitude
        mu = user["mu"][strategy]
        user["mu"][strategy] = mu * (1.0 - config.NEG_MU_SHRINK * s)

        # shrink precision -> increases covariance (more uncertainty)
        fac = max(0.15, 1.0 - config.NEG_SINV_SHRINK * s)
        user["sigma_inv"][strategy] = user["sigma_inv"][strategy] * fac

    def select(self, uid: str, message: str, *,
               force_explore: bool = False,
               neg_strength: float = 0.0,
               explicit_strategy: str | None = None):
        """Return (chosen, scores, x, prev_strategy)."""
        user = self.get_user(uid)
        prev = user.get("last_strategy")
        x = self.featurize(message, user)

        # One-time override: honor the upfront user-selected format once,
        # then return to adaptive Thompson Sampling on later turns.
        pending = user.get("pending_strategy")
        if pending in self.global_mu:
            user["pending_strategy"] = None
            return pending, {k: 0.0 for k in config.STRATEGY_NAMES}, x, prev

        # If user explicitly asked for a format, obey immediately.
        # Hard-lock disables exploration, but should NOT block an explicit request like
        # "compare X vs Y" or "put it in a table".
        locked = user.get("locked_strategy")
        if locked in self.global_mu:
            force_explore = False
            if explicit_strategy is None:
                explicit_strategy = locked

        # Corrective exploration: damp posterior on prev to avoid getting stuck.
        if force_explore and prev:
            self._damp_posterior(user, prev, neg_strength)

        # Build TS scores
        temp = config.TS_TEMPERATURE * (config.EXPLORE_TEMP_BOOST if force_explore else 1.0)
        scores = {}
        for k in config.STRATEGY_NAMES:
            sigma = np.linalg.inv(user["sigma_inv"][k])
            beta = np.random.multivariate_normal(user["mu"][k], sigma * temp)
            scores[k] = float(sigmoid(x @ beta))

        # Soft preference nudges (does NOT lock)
        pref_boost = 0.06
        for s in user.get("prefs", set()):
            if s in scores:
                scores[s] = float(min(0.999, scores[s] + pref_boost))

        # Strong anti-repeat penalty when exploring (unless explicit override)
        if force_explore and prev and prev in scores and explicit_strategy is None and len(config.STRATEGY_NAMES) > 1:
            scores[prev] = float(scores[prev] - config.EXPLORE_SCORE_PENALTY)

        # Choose
        # Explicit user request (e.g., "compare", "table", "graph") should win even if the user
        # previously hard-locked a default style. The lock only disables exploration.
        if explicit_strategy in config.STRATEGY_NAMES:
            chosen = explicit_strategy
        elif locked in self.global_mu and locked in config.STRATEGY_NAMES:
            chosen = locked
        else:
            chosen = max(scores, key=scores.get)

        return chosen, scores, x, prev

    def update(self, uid: str, strategy: str, x: np.ndarray, reward: float):
        user   = self.get_user(uid)
        if strategy not in user.get("mu", {}) or strategy not in user.get("sigma_inv", {}):
            # Strategy might have been hard-deleted after selection; ignore update safely.
            return
        mu_old = user["mu"][strategy]
        si_old = user["sigma_inv"][strategy]
        r_hat  = sigmoid(float(x @ mu_old))
        w      = r_hat * (1 - r_hat)

        si_new = config.GAMMA * si_old + np.outer(x, x) * w + config.LAMBDA * np.eye(config.D)
        s_new  = np.linalg.inv(si_new)
        user["mu"][strategy]        = mu_old + s_new @ x * (reward - r_hat)
        user["sigma_inv"][strategy] = si_new
        user["reward_log"].append((strategy, reward))

        # Global update
        gmu    = self.global_mu[strategy]
        gsi    = self.global_sinv[strategy]
        gr_hat = sigmoid(float(x @ gmu))
        gsi_n  = config.GAMMA*gsi + np.outer(x,x)*gr_hat*(1-gr_hat)*config.ALPHA_G + config.LAMBDA*np.eye(config.D)
        gs_n   = np.linalg.inv(gsi_n)
        self.global_mu[strategy]   = gmu + gs_n @ x * (reward - gr_hat) * config.ALPHA_G
        self.global_sinv[strategy] = gsi_n
        self.global_n += 1

    def apply_preferences(self, uid: str, strategy_names, *, lock: bool = False):
        """Apply soft preferences and optional hard lock.

        Behavior:
        - if lock=True and exactly one strategy is chosen: always use that strategy
        - if lock=False and exactly one strategy is chosen: force it ONCE on the next turn,
          then return to adaptive TS with preference bias
        """
        user = self.get_user(uid)
        chosen = [s for s in (strategy_names or []) if s in config.STRATEGY_NAMES]
        user["prefs"] = set(chosen)

        # hard lock only if explicitly requested
        user["locked_strategy"] = chosen[0] if (lock and len(chosen) == 1) else None

        # one-turn override for the very next assistant response
        user["pending_strategy"] = chosen[0] if (not lock and len(chosen) == 1) else None

        # warm-start: nudge mean for preferred arms
        for s in user["prefs"]:
            user["mu"][s] = user["mu"][s] + 0.5

    def posterior_summary(self, mu_dict, sinv_dict, x=None):
        if x is None:
            x = np.ones(config.D) * 0.5
        return {k: {"r": round(float(sigmoid(x @ mu_dict[k])), 4),
                    "u": round(mean_uncertainty(sinv_dict[k]), 4)}
                for k in config.STRATEGY_NAMES}

    def user_posterior(self, uid: str, x=None):
        u = self.get_user(uid)
        return self.posterior_summary(u["mu"], u["sigma_inv"], x)

    def global_posterior(self, x=None):
        return self.posterior_summary(self.global_mu, self.global_sinv, x)

    def reset_user(self, uid: str):
        self.users.pop(uid, None)


engine = BayesianEngine()
USERB_ID = "__user_b__"
engine.get_user(USERB_ID)

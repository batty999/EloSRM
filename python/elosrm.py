from __future__ import annotations
import math
from typing import List
import numpy as np
from numba import njit

#
# EloSRM rating system for TopCoder SRM
# https://github.com/batty999/EloSRM
# (c) 2019-2026 Fred Batty
#

def _prepare_arrays(results, cls):
    """Extract player data into NumPy arrays for JIT compilation."""
    n = len(results)

    ratings = np.empty(n, dtype=np.float64)
    scores = np.empty(n, dtype=np.float64)
    num_ratings = np.empty(n, dtype=np.float64)
    recent_rounds = np.empty(n, dtype=np.float64)

    for i, r in enumerate(results):
        p = r.player
        ratings[i] = p.rating
        num_ratings[i] = p.num_ratings
        recent_rounds[i] = p.recent_rounds
        scores[i] = r.score

    # Vectorized exp
    rating_exp = np.exp(ratings * cls.ELO_SCALE)

    return rating_exp, scores, num_ratings, recent_rounds

@njit(fastmath=True)
def _rate_one_division(
        rates,        # np.ndarray   (10 ** (rating / 400))
        scores,       # np.ndarray   (float)
        nr,           # np.ndarray   (player.num_ratings)
        recent,       # np.ndarray   (player.recent_rounds)
        M, B, W1, C, gamma, K):
    """
    Core rating calculation compiled to native code.
    
    Returns:
        delta: Rating change for each player
        perf: Performance for each player
    """
    n = rates.shape[0]
    delta = np.empty(n, dtype=np.float64)
    perf = np.empty(n, dtype=np.float64)

    for i in range(n):
        ri = rates[i]
        si = scores[i]

        erank = 1.0     # Expected rank
        arank = 1.0     # Actual rank
        mu = 1.0
        var = 1.0

        for j in range(n):
            if j == i:
                continue

            rj = rates[j]
            sj = scores[j]

            wj = rj / (ri + rj)     # win probability

            mu  += wj
            var += wj * (1.0 - wj)

            if si == sj:
                erank += 0.5
                arank += 0.5
            else:
                erank += wj
                arank += 1.0 if si < sj else 0.0

        # Calculate performance
        perf_i = math.log2(erank / arank)
        perf1  = var / mu

        pa = perf_i * M / (M + abs(perf_i))
        pa += B * perf1

        # Weight factors
        ef = (1.0 + nr[i] * W1) ** 0.5
        cf = 1.0 + C * perf1
        ff = recent[i] ** gamma

        w = ef * cf * ff

        delta[i] = K * pa / w
        perf[i]  = perf_i

    return delta, perf

class EloSRM:
    """Python implementation of EloSRM rating system"""
    
    # Constants
    VERSION = 7
    ELO_SCALE = math.log(10) / 400
    K0 = 400 * math.log(2) / math.log(10)
    R0 = 1200.0                 # Initial rating

    # System parameters
    K = 648.3147599935407       # Base K-factor
    C = 3.8884120557511483      # Competition factor
    M = 4.44015774770823        # Max performance
    B = 41.84027892146161       # Performance bonus
    W1 = 0.20058285315948782    # Experience weight
    D = 112.17636272246507      # Recent period (days)
    gamma = 0.4599945496035186  # Frequency factor
    G = 50.67198262989024       # Inflation per year
    
    # Runtime computed values
    t0: float = 0.0             # First round
    R_init: float = R0          # adjusted for inflation
    B2: float = 0.0
    lambda_val: float = 0.0
    G_sec: float = 0.0
    
    class Player:
        """Represents a player's rating state."""
        __slots__ = ['num_ratings', 'rating', 'recent_rounds', 'last_round']

        def __init__(self):
            self.num_ratings = 0
            self.rating = EloSRM.R_init
            self.recent_rounds = 0
            self.last_round = 0
        
        def __repr__(self):
            return f"Player(rating={self.rating:.1f}, num_ratings={self.num_ratings})"
    
    class Result:
        """Represents a player's result in a round."""
        __slots__ = ['player', 'score', 'delta_r', 'perf']

        def __init__(self, player: EloSRM.Player, score: float):
            self.player = player
            self.score = score
            self.delta_r = 0.0
            self.perf = 0.0
        
        def __repr__(self):
            return f"Result(score={self.score}, delta={self.delta_r:.1f})"

    @classmethod
    def rate_division(cls, results: List[Result]) -> None:
        """Calculate rating changes for all players in a division."""
        if not results:
            return

        # Extract data into NumPy arrays
        rates, scores, nr, recent = _prepare_arrays(results, cls)

        # Run JIT-compiled calculation
        delta, perf = _rate_one_division(
            rates, scores, nr, recent,
            cls.M, cls.B2, cls.W1, cls.C, cls.gamma, cls.K
        )

        # Apply results
        for i, r in enumerate(results):
            r.delta_r = delta[i]
            r.perf    = perf[i]

    @classmethod
    def rate_round(cls, results: List[Result], round_time: float) -> None:
        """Rate a complete round, updating all player ratings."""
        # Adjust initial rating for inflation
        cls.R_init = cls.R0 + cls.G_sec * (round_time - cls.t0)
        
        # Update player statistics before rating
        for result in results:
            pi = result.player
            
            if pi.num_ratings == 0:
                pi.rating = cls.R_init
                pi.recent_rounds = 1.0
            else:
                t_diff = round_time - pi.last_round
                decay = math.exp(cls.lambda_val * t_diff)
                pi.recent_rounds = 1.0 + pi.recent_rounds * decay

        # Calculate rating changes
        cls.rate_division(results)

        # Apply rating changes
        for result in results:
            pi = result.player
            pi.num_ratings += 1
            pi.rating += result.delta_r
            pi.last_round = round_time

    @classmethod
    def init(cls, first_round_time: float) -> None:
        """Initialize the rating system"""
        cls.t0 = first_round_time
        cls.R_init = cls.R0
        cls.G_sec = cls.G / (365.25 * 24 * 3600)
        cls.B2 = cls.B / cls.K0
        cls.lambda_val = -math.log(2) / (cls.D * 24 * 3600)

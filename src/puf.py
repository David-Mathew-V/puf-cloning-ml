"""Arbiter / XOR Arbiter PUF simulator using the additive delay model."""
import numpy as np


def random_challenges(N, n, rng):
    return rng.integers(0, 2, size=(N, n), dtype=np.int8)


def parity_features(challenges, dtype=np.float32):
    # phi_i = prod_{j>=i} (1 - 2 c_j), plus a constant 1 at the end
    signs = (1 - 2 * challenges.astype(np.int8)).astype(dtype)
    phi = np.cumprod(signs[:, ::-1], axis=1)[:, ::-1]
    ones = np.ones((challenges.shape[0], 1), dtype=dtype)
    return np.ascontiguousarray(np.hstack([phi, ones]), dtype=dtype)


class XORArbiterPUF:
    # k=1 is a normal arbiter PUF. noisiness is relative to the std of the delay difference.
    def __init__(self, n, k=1, noisiness=0.0, seed=0):
        self.n, self.k, self.noisiness = n, k, noisiness
        self._rng = np.random.default_rng(seed)
        self.weights = self._rng.normal(0.0, 1.0, size=(k, n + 1))
        self._noise_std = noisiness * np.sqrt(n + 1)

    def delay_differences(self, phi, noisy=True):
        d = phi @ self.weights.T
        if noisy and self._noise_std > 0:
            d = d + self._rng.normal(0.0, self._noise_std, size=d.shape)
        return d

    def eval_phi(self, phi, noisy=True):
        s = np.sign(self.delay_differences(phi, noisy))
        s[s == 0] = 1
        return np.prod(s, axis=1).astype(np.int8)  # XOR of the chains in +/-1 form

    def eval(self, challenges, noisy=True):
        return self.eval_phi(parity_features(challenges), noisy)

    def reliability(self, n_challenges=20000, seed=123):
        rng = np.random.default_rng(seed)
        phi = parity_features(random_challenges(n_challenges, self.n, rng))
        return float(np.mean(self.eval_phi(phi) == self.eval_phi(phi)))


def to_bits(r):
    return (r < 0).astype(np.int64)


def make_crp_dataset(puf, N, seed, noisy=True):
    rng = np.random.default_rng(seed)
    c = random_challenges(N, puf.n, rng)
    phi = parity_features(c)
    y = to_bits(puf.eval_phi(phi, noisy=noisy))
    return c, phi, y

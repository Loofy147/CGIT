"""
Cognitive Grammar Induction Test (CGIT) — implementation.

Levels:
  Data -> Representation -> Operator Sequence -> Cognitive Grammar

Everything here is domain-agnostic except the domain generators themselves.
Operators, the situation encoder, and the grammar/evolution machinery never
see a domain label — only arrays.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

OPS = ["DISTINGUISH", "RELATE", "COMPARE", "COMPRESS", "PREDICT", "ABSTRACT", "SPECIALIZE", "SIMULATE", "INTERACT"]
OP_IDX = {o: i for i, o in enumerate(OPS)}
MAX_DIM = 60          # safety cap so concatenating operators can't blow up cost
SIT_DIM = 7            # situation signature length

# ---------------------------------------------------------------- domains --
def _std(X):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    sd[sd < 1e-8] = 1e-8
    return (X - mu) / sd

def make_physics(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(70, 140)
    amp = rng.uniform(0.5, 2.0, n); phase = rng.uniform(0, 2*np.pi, n)
    damping = rng.uniform(0.01, 0.2, n); freq = rng.uniform(0.8, 1.2, n)
    samp_t = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    X = np.array([amp[i]*np.exp(-damping[i]*samp_t)*np.cos(freq[i]*samp_t+phase[i]) for i in range(n)])
    X += rng.normal(0, 0.03, X.shape)
    extra = np.column_stack([amp*rng.normal(1, 0.02, n), damping*rng.normal(1, 0.05, n), freq*rng.normal(1, 0.02, n)])
    X = np.hstack([X, extra])
    y = amp*np.exp(-damping*5.0)*np.cos(freq*5.0+phase)   # extrapolated future amplitude
    return _std(X), y, "reg"

def make_software(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(70, 140)
    fan_in = rng.poisson(4, n).astype(float); fan_out = rng.poisson(3, n).astype(float)
    churn = rng.exponential(1.0, n); coverage = rng.uniform(0, 1, n)
    hist_fail = rng.beta(1.5, 6, n)
    logit = 1.4*hist_fail + 0.15*fan_in + 0.1*churn - 1.3*coverage - 1.0
    logit += rng.normal(0, 0.4, n)
    p = 1/(1+np.exp(-logit))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    # inject a block of mislabeled/contradictory rows (flaky signal)
    flip = rng.choice(n, size=max(1, int(0.12*n)), replace=False)
    y[flip] = 1 - y[flip]
    X = np.column_stack([fan_in, fan_out, churn, coverage, hist_fail, fan_in*churn])
    return _std(X), y, "clf"

def make_biology(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(70, 140)
    expr = rng.normal(0, 1, (n, 6))
    regulator = expr[:, 0]
    feedback_term = 0.3*regulator  # weak internal feedback loop
    fitness = 0.5*expr[:, 1] - 0.4*expr[:, 2] + 0.3*expr[:, 3]*expr[:, 4] + feedback_term
    fitness += rng.normal(0, 0.5, n)
    X = np.column_stack([expr, np.roll(feedback_term, 1)])  # lagged feedback visible as a feature
    order = np.argsort(rng.uniform(size=n))  # weak temporal ordering (generations)
    X, y = X[order], fitness[order]
    return _std(X), y, "reg"

def make_economics(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(80, 150)
    price = np.zeros(n); price[0] = 0.0
    shocks = rng.normal(0, 1, n)
    for t in range(1, n):
        price[t] = 0.85*price[t-1] + 0.4*shocks[t] + 0.2*np.tanh(price[t-1])  # strong feedback loop
    lag1 = np.roll(price, 1); lag2 = np.roll(price, 2); vol = np.roll(np.abs(shocks), 1)
    X = np.column_stack([lag1, lag2, vol, shocks])
    y = price  # predict current level from lagged info
    return _std(X[2:]), y[2:], "reg"

def make_robotics(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(80, 150)
    state = np.zeros((n, 3))
    action = rng.normal(0, 1, (n, 2))
    for t in range(1, n):
        state[t] = 0.9*state[t-1] + 0.5*np.array([action[t-1, 0], action[t-1, 1], action[t-1].sum()])
        state[t] += rng.normal(0, 0.15, 3)  # sensor noise
    err = np.linalg.norm(state[1:] - np.roll(state, -1, axis=0)[1:], axis=1)
    y = (err < np.median(err)).astype(int)  # "success" = low tracking error next step
    X = np.column_stack([state, action])[1:]
    return _std(X), y, "clf"

def make_social(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(80, 150)
    influence = rng.normal(0, 1, n)
    peer_pressure = rng.normal(0, 1, n)
    conflicting_signal = rng.normal(0, 1, n)
    adopt_prob_logit = 0.9*influence + 0.7*peer_pressure - 0.5*conflicting_signal
    # feedback: adoption of "neighbors" (proxy = sorted order neighbor) raises own logit
    order = np.argsort(influence)
    neighbor_effect = np.zeros(n)
    neighbor_effect[order[1:]] = 0.4*np.sign(adopt_prob_logit[order[:-1]])
    adopt_prob_logit += neighbor_effect
    p = 1/(1+np.exp(-adopt_prob_logit))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    # inject polarized/contradictory subgroup
    contra = rng.choice(n, size=max(1, int(0.15*n)), replace=False)
    conflicting_signal[contra] *= -3
    X = np.column_stack([influence, peer_pressure, conflicting_signal, neighbor_effect])
    return _std(X), y, "clf"

DOMAIN_GENERATORS = {
    "physics": make_physics, "software": make_software, "biology": make_biology,
    "economics": make_economics, "robotics": make_robotics, "social": make_social,
}
TRAIN_DOMAINS = ["physics", "software", "biology"]
TEST_DOMAINS = ["economics", "robotics", "social"]

# --------------------------------------------------------- situation enc. --
def situation_signature(X, y, task):
    n, d = X.shape
    w = max(3, n // 15)
    kernel = np.ones(w) / w
    Xs = np.vstack([np.convolve(X[:, j], kernel, mode="same") for j in range(d)]).T
    noise = np.mean(np.var(X - Xs, axis=0)) / (np.mean(np.var(X, axis=0)) + 1e-8)
    noise = min(noise, 1.0)

    def safe_corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8: return 0.0
        c = np.corrcoef(a, b)[0, 1]
        return 0.0 if np.isnan(c) else c

    temp_dep = np.mean([abs(safe_corr(X[:-1, j], X[1:, j])) for j in range(d)])
    n_norm = np.tanh(n / 100)
    half = n // 2
    corr1 = np.array([safe_corr(X[:half, j], y[:half]) for j in range(d)])
    corr2 = np.array([safe_corr(X[half:, j], y[half:]) for j in range(d)])
    contradiction = np.mean((np.sign(corr1) != np.sign(corr2)) & (np.abs(corr1) > 0.05) & (np.abs(corr2) > 0.05))
    if task == "reg":
        uncertainty = np.tanh(np.std(y) / (abs(np.mean(y)) + 1e-6))
    else:
        p = np.bincount(y.astype(int)) / len(y); p = p[p > 0]
        uncertainty = float(-(p*np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
    fb = np.mean([abs(safe_corr(y[:-1], X[1:, j])) for j in range(d)]) if n > 2 else 0.0
    Xc = X - X.mean(0)
    try:
        sv = np.linalg.svd(Xc, compute_uv=False); var = sv**2
        opt_pressure = float(var[:3].sum() / (var.sum() + 1e-8))
    except Exception:
        opt_pressure = 0.5
    s = np.array([noise, temp_dep, n_norm, contradiction, uncertainty, fb, opt_pressure])
    return np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)

# --------------------------------------------------------------- operators--
def _kmeans_centroids(R, k):
    k = max(1, min(k, R.shape[0]-1, 8))
    if k < 2: return R * 0 + R.mean(0, keepdims=True)
    km = KMeans(n_clusters=k, n_init=1, random_state=0).fit(R)
    return km.cluster_centers_[km.labels_]

def op_distinguish(R):
    k = max(2, R.shape[0] // 25)  # coarse split -> contrastive residual
    return np.hstack([R, R - _kmeans_centroids(R, k)])

def op_relate(R):
    sim = cosine_similarity(R); np.fill_diagonal(sim, -np.inf)
    k = min(5, R.shape[0]-1)
    idx = np.argsort(-sim, axis=1)[:, :k]
    neigh = R[idx].mean(axis=1)
    return np.hstack([R, neigh])

def op_compare(R):
    gm = R.mean(0, keepdims=True)
    sim = cosine_similarity(R); np.fill_diagonal(sim, -np.inf)
    nn = np.argmax(sim, axis=1)
    return np.hstack([R, R - gm, R - R[nn]])

def op_compress(R):
    d = R.shape[1]
    k = max(1, int(np.ceil(d*0.55)))
    if k >= d: return R
    k = min(k, R.shape[0]-1)
    if k < 1: return R
    return PCA(n_components=k, random_state=0).fit_transform(R)

def op_predict(R):
    t = np.arange(R.shape[0]).reshape(-1, 1).astype(float)
    trend = Ridge(alpha=1.0).fit(t, R).predict(t)
    return np.hstack([R, trend])

def op_abstract(R):
    k = max(2, R.shape[0] // 20)   # fine-ish clusters -> generalize by replacing w/ centroid
    return _kmeans_centroids(R, k)

def op_specialize(R):
    k = max(3, R.shape[0] // 10)   # finer clusters than DISTINGUISH -> reinject specific detail
    return np.hstack([R, R - _kmeans_centroids(R, k)])

def op_simulate(R):
    sim = cosine_similarity(R); np.fill_diagonal(sim, 0); sim = np.clip(sim, 0, None)
    rs = sim.sum(1, keepdims=True); rs[rs == 0] = 1
    A = sim / rs
    Rt = R.copy()
    for _ in range(2):
        Rt = 0.7*Rt + 0.3*(A @ Rt)
    return np.hstack([R, Rt])

def op_interact(R):
    n, d = R.shape
    if d < 2:
        return R
    cols = []
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(R[:, i] * R[:, j])
    if len(cols) == 0:
        return R
    inter = np.column_stack(cols)
    return np.hstack([R, inter])

OP_FUNCS = [op_distinguish, op_relate, op_compare, op_compress, op_predict, op_abstract, op_specialize, op_simulate, op_interact]

def run_program(X, op_seq):
    R = X.copy()
    for op in op_seq:
        R = OP_FUNCS[op](R)
        if R.shape[1] > MAX_DIM:
            k = min(MAX_DIM, R.shape[0]-1, R.shape[1]-1)
            if k >= 1:
                R = PCA(n_components=k, random_state=0).fit_transform(R)
    return R

# ------------------------------------------------------------------ eval --
def evaluate_representation(R, y, task, seed=0, want_true_fragility=False, X_for_noise=None, op_seq=None):
    n = R.shape[0]
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed)
    Rn = _std(R)
    if task == "reg":
        model = Ridge(alpha=1.0).fit(Rn[tr], y[tr])
        pred = model.predict(Rn[te])
        perf = max(0.0, r2_score(y[te], pred))
    else:
        if len(np.unique(y[tr])) < 2:
            perf = 0.0
        else:
            model = LogisticRegression(max_iter=300).fit(Rn[tr], y[tr])
            pred = model.predict(Rn[te])
            acc = accuracy_score(y[te], pred)
            chance = max(np.mean(y == 1), np.mean(y == 0))
            perf = max(0.0, (acc - chance) / (1 - chance + 1e-8))
    fragility = None
    if want_true_fragility and X_for_noise is not None and op_seq is not None:
        rng = np.random.default_rng(seed+999)
        Xn = X_for_noise + rng.normal(0, 0.15*np.std(X_for_noise), X_for_noise.shape)
        R2 = run_program(Xn, op_seq)
        perf2 = evaluate_representation(R2, y, task, seed=seed)["perf"]
        fragility = abs(perf - perf2)
    return {"perf": perf, "fragility": fragility}

def fitness(X, y, task, op_seq, lambda_cost=1.0, seed=0, cheap_fragility=True):
    R = run_program(X, op_seq)
    ev = evaluate_representation(R, y, task, seed=seed)
    cost = 0.03*len(op_seq) + 0.0008*R.shape[1]
    if cheap_fragility:
        cov = np.std(R) / (np.mean(np.abs(R)) + 1e-6)
        frag_proxy = np.clip(0.1*cov, 0, 1)
    else:
        frag_proxy = 0.0
    F = ev["perf"] - lambda_cost*cost - 0.5*frag_proxy
    return F, ev["perf"], cost, R.shape[1]

# --------------------------------------------------------------- grammar --
class Rule:
    __slots__ = ("prototype", "seq")
    def __init__(self, prototype, seq):
        self.prototype = np.array(prototype, dtype=float)
        self.seq = list(seq)
    def copy(self):
        return Rule(self.prototype.copy(), list(self.seq))

class Grammar:
    def __init__(self, rules):
        self.rules = rules
    def match(self, s):
        if len(self.rules) == 1:
            return self.rules[0].seq, 0
        d = [np.linalg.norm(r.prototype - s) for r in self.rules]
        i = int(np.argmin(d))
        return self.rules[i].seq, i
    def copy(self):
        return Grammar([r.copy() for r in self.rules])

def random_seq(rng, minlen=1, maxlen=4):
    L = rng.integers(minlen, maxlen+1)
    return [int(x) for x in rng.integers(0, len(OPS), L)]

def random_grammar(rng, proto_pool, k=None):
    k = k or rng.integers(2, 5)
    rules = []
    for _ in range(k):
        proto = proto_pool[rng.integers(0, len(proto_pool))] + rng.normal(0, 0.05, SIT_DIM)
        rules.append(Rule(proto, random_seq(rng)))
    return Grammar(rules)

def seeded_initial_population(pop_size, rng, proto_pool):
    pop = []
    pop.append(Grammar([Rule(proto_pool[0], [OP_IDX["COMPRESS"], OP_IDX["PREDICT"]])]))          # "Always" G1
    pop.append(Grammar([Rule(proto_pool[0], [OP_IDX["DISTINGUISH"], OP_IDX["RELATE"]])]))          # "Always" G2
    pop.append(random_grammar(rng, proto_pool))                                                    # random G3
    while len(pop) < pop_size:
        pop.append(random_grammar(rng, proto_pool))
    return pop

def mutate(g, rng, proto_pool):
    g = g.copy()
    for r in g.rules:
        if rng.random() < 0.3:
            r.prototype = r.prototype + rng.normal(0, 0.12, SIT_DIM)
        if rng.random() < 0.3:
            seq = r.seq[:]
            act = rng.choice(["swap", "insert", "delete", "sub"])
            if act == "swap" and len(seq) >= 1:
                seq[rng.integers(0, len(seq))] = int(rng.integers(0, len(OPS)))
            elif act == "insert" and len(seq) < 4:
                seq.insert(int(rng.integers(0, len(seq)+1)), int(rng.integers(0, len(OPS))))
            elif act == "delete" and len(seq) > 1:
                del seq[int(rng.integers(0, len(seq)))]
            elif act == "sub":
                i = int(rng.integers(0, len(seq)))
                seq[i] = int(rng.integers(0, len(OPS)))
            r.seq = seq
    if rng.random() < 0.12 and len(g.rules) < 6:
        proto = proto_pool[rng.integers(0, len(proto_pool))] + rng.normal(0, 0.08, SIT_DIM)
        g.rules.append(Rule(proto, random_seq(rng)))
    if rng.random() < 0.08 and len(g.rules) > 1:
        del g.rules[rng.integers(0, len(g.rules))]
    return g

def crossover(g1, g2, rng):
    rules = [r.copy() for r in g1.rules if rng.random() < 0.5] + [r.copy() for r in g2.rules if rng.random() < 0.5]
    if not rules:
        rules = [g1.rules[0].copy()]
    rng.shuffle(rules)
    return Grammar(rules[:6])

def tournament(pop, fits, rng, k=3):
    idx = rng.integers(0, len(pop), k)
    best = idx[np.argmax([fits[i] for i in idx])]
    return pop[best]

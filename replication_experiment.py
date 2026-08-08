import numpy as np, json, time
from scipy import stats
from collections import defaultdict, Counter
from cgit_lib import (
    DOMAIN_GENERATORS, TRAIN_DOMAINS, TEST_DOMAINS, OPS, OP_IDX,
    situation_signature, fitness, seeded_initial_population, mutate, crossover, tournament,
    Rule, Grammar,
)

t0 = time.time()
ALL_DOMAINS = TRAIN_DOMAINS + TEST_DOMAINS
v2 = json.load(open("/home/claude/v2_results.json"))

def build_pool(domain, seeds):
    pool = []
    for s in seeds:
        X, y, task = DOMAIN_GENERATORS[domain](s)
        sig = situation_signature(X, y, task)
        pool.append(dict(X=X, y=y, task=task, sig=sig, seed=s))
    return pool

# candidate per domain: the validated winner if non-empty, else the runner-up
# that the archive ranked best but lost to raw on the single earlier holdout
CANDIDATES = {}
for dom in ALL_DOMAINS:
    validated_seq = v2["validated"][dom]["seq"]
    if validated_seq:
        CANDIDATES[dom] = validated_seq
    else:
        tiers = v2["tiers"][dom]
        best_len = max(tiers, key=lambda L: tiers[L]["best_perf"])
        CANDIDATES[dom] = tiers[best_len]["best_seq"]
    print(f"{dom:10s} candidate under replication test: {[OPS[i] for i in CANDIDATES[dom]]}")

# ================================================================ PART A =
# Fixed candidate vs raw, N=25 fresh independent draws per domain, paired
# (same data + same OOF split -> isolates the operator-sequence effect),
# with bootstrap + t-based 95% CI and a paired significance test.
print("\n=== PART A: replication of fixed candidate vs raw (N=25 fresh draws/domain) ===")
N_REPLICATE = 25
partA = {}
for i, dom in enumerate(ALL_DOMAINS):
    base = 5000 + i*100  # fresh seed range, never used anywhere else in this project
    diffs = []
    for j in range(N_REPLICATE):
        s = base + j
        X, y, task = DOMAIN_GENERATORS[dom](s)
        _, perf_cand, _, _ = fitness(X, y, task, CANDIDATES[dom], seed=s)
        _, perf_raw, _, _ = fitness(X, y, task, [], seed=s)
        diffs.append(perf_cand - perf_raw)
    diffs = np.array(diffs)
    mean_d = diffs.mean(); sem = stats.sem(diffs)
    ci = stats.t.interval(0.95, len(diffs)-1, loc=mean_d, scale=sem) if sem > 0 else (mean_d, mean_d)
    tstat, pval = stats.ttest_1samp(diffs, 0.0)
    try:
        wstat, wpval = stats.wilcoxon(diffs)
    except ValueError:
        wpval = float('nan')
    win_rate = float(np.mean(diffs > 0))
    partA[dom] = dict(mean_diff=float(mean_d), ci95=[float(ci[0]), float(ci[1])],
                       t_pvalue=float(pval), wilcoxon_pvalue=float(wpval), win_rate=win_rate,
                       n=N_REPLICATE, candidate=[OPS[k] for k in CANDIDATES[dom]])
    sig = "SIGNIFICANT" if ci[0] > 0 else ("SIG. NEGATIVE" if ci[1] < 0 else "not significant (CI crosses 0)")
    print(f"  {dom:10s} mean_diff={mean_d:+.4f}  95% CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]  "
          f"win_rate={win_rate:.2f}  p(t)={pval:.4f}  -> {sig}")

# ================================================================ PART B =
# Search replicability: rerun the FULL nested-OOS search + validated-selection
# pipeline (v2 methodology) on physics / social / software, with independent
# evolutionary seeds AND independent fresh data draws each time, to check
# whether the SEARCH reliably rediscovers a winner (not just whether a fixed
# sequence, once found, holds up).
print("\n=== PART B: independent search replicates (physics, social, software) ===")

def perf_only_fitness_evolve(g, evolve_pool):
    total = 0.0
    for rec in evolve_pool:
        seq, _ = g.match(rec["sig"])
        F, perf, cost, dim = fitness(rec["X"], rec["y"], rec["task"], seq, lambda_cost=0.0, seed=rec["seed"])
        total += F
    return total / len(evolve_pool)

def log_archive(g, archive_eval_pool, archive):
    for rec in archive_eval_pool:
        seq, _ = g.match(rec["sig"])
        F, perf, cost, dim = fitness(rec["X"], rec["y"], rec["task"], seq, lambda_cost=0.0, seed=rec["seed"])
        archive[tuple(seq)].append((perf, cost))

def evolve_nested(evolve_pool, archive_eval_pool, proto_pool, pop_size=16, generations=14, seed=0):
    rng = np.random.default_rng(seed)
    archive = defaultdict(list)
    pop = seeded_initial_population(pop_size, rng, proto_pool)
    for gen in range(generations):
        fits = [perf_only_fitness_evolve(g, evolve_pool) for g in pop]
        for g in pop:
            log_archive(g, archive_eval_pool, archive)
        order = np.argsort(fits)[::-1]
        new_pop = [pop[order[0]].copy(), pop[order[1]].copy()]
        while len(new_pop) < pop_size:
            p1 = tournament(pop, fits, rng); p2 = tournament(pop, fits, rng)
            child = crossover(p1, p2, rng) if rng.random() < 0.6 else p1.copy()
            child = mutate(child, rng, proto_pool)
            new_pop.append(child)
        pop = new_pop
    return archive

N_SEARCH_SEEDS = 5
partB = {}
for dom in ["physics", "social", "software"]:
    replicate_results = []
    for rep in range(N_SEARCH_SEEDS):
        offset = 8000 + hash((dom, rep)) % 1000  # fresh, non-overlapping data per replicate
        evolve_pool = build_pool(dom, range(offset, offset+8))
        archive_eval_pool = build_pool(dom, range(offset+20, offset+26))
        final_holdout = build_pool(dom, range(offset+40, offset+44))
        proto_pool = np.array([r["sig"] for r in evolve_pool])
        archive = evolve_nested(evolve_pool, archive_eval_pool, proto_pool, seed=rep)
        agg = {seq: dict(perf=float(np.mean([v[0] for v in vals])), cost=float(np.mean([v[1] for v in vals])),
                          seqlen=len(seq)) for seq, vals in archive.items()}
        by_len = defaultdict(list)
        for seq, rec in agg.items():
            by_len[rec["seqlen"]].append((rec["perf"], seq))
        candidates = [max(v, key=lambda x: x[0])[1] for v in by_len.values()]
        candidates.append(tuple())
        best_seq, best_score = None, -1e9
        for seq in candidates:
            perfs = [fitness(rec["X"], rec["y"], rec["task"], list(seq), seed=rec["seed"])[1] for rec in final_holdout]
            costs = [fitness(rec["X"], rec["y"], rec["task"], list(seq), seed=rec["seed"])[2] for rec in final_holdout]
            score = np.mean(perfs) - 1.0*np.mean(costs)
            if score > best_score:
                best_score, best_seq, best_perf = score, seq, float(np.mean(perfs))
        raw_perf = float(np.mean([fitness(rec["X"], rec["y"], rec["task"], [], seed=rec["seed"])[1] for rec in final_holdout]))
        beats_raw = best_perf > raw_perf + 1e-9
        replicate_results.append(dict(seq=[OPS[i] for i in best_seq], perf=best_perf, raw_perf=raw_perf, beats_raw=beats_raw))
        print(f"  {dom:10s} rep{rep}: winner={str([OPS[i] for i in best_seq]):25}  perf={best_perf:.3f}  raw={raw_perf:.3f}  beats_raw={beats_raw}")
    n_beats = sum(r["beats_raw"] for r in replicate_results)
    partB[dom] = dict(replicates=replicate_results, n_beats_raw=n_beats, n_total=N_SEARCH_SEEDS)
    print(f"  -> {dom}: search beat raw in {n_beats}/{N_SEARCH_SEEDS} independent replicates")

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed:.1f}s")

out = dict(candidates={d: [OPS[i] for i in s] for d, s in CANDIDATES.items()},
           partA=partA, partB=partB, elapsed_seconds=elapsed)
with open("/home/claude/replication_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved /home/claude/replication_results.json")

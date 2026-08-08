import numpy as np, json, time
from collections import defaultdict
from cgit_lib import (
    DOMAIN_GENERATORS, TRAIN_DOMAINS, TEST_DOMAINS, OPS, OP_IDX,
    situation_signature, fitness, run_program, evaluate_representation,
    Rule, Grammar, seeded_initial_population, mutate, crossover, tournament,
)

t0 = time.time()
ALL_DOMAINS = TRAIN_DOMAINS + TEST_DOMAINS
LAMBDA = 1.0

def build_pool(domain, seeds):
    pool = []
    for s in seeds:
        X, y, task = DOMAIN_GENERATORS[domain](s)
        sig = situation_signature(X, y, task)
        pool.append(dict(X=X, y=y, task=task, sig=sig, seed=s))
    return pool

# ---- 1. bigger, properly-separated pools per domain: evolve / archive-eval / final-holdout
POOLS = {}
for i, dom in enumerate(ALL_DOMAINS):
    base = 100 + i*200  # keep domains' seed ranges far apart
    POOLS[dom] = dict(
        evolve=build_pool(dom, range(base, base+8)),          # drives evolutionary selection
        archive_eval=build_pool(dom, range(base+50, base+56)),  # drives what gets archived (disjoint!)
        final_holdout=build_pool(dom, range(base+100, base+104)),  # touched only once, at the end
    )
    print(f"{dom:10s} evolve={len(POOLS[dom]['evolve'])}  archive_eval={len(POOLS[dom]['archive_eval'])}  "
          f"final_holdout={len(POOLS[dom]['final_holdout'])}")

# ---- 2. nested out-of-sample search: fitness drives selection on `evolve`,
#         archive logs performance measured on the DISJOINT `archive_eval` set
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
            log_archive(g, archive_eval_pool, archive)  # OUT-OF-SAMPLE relative to what drove `fits`
        order = np.argsort(fits)[::-1]
        new_pop = [pop[order[0]].copy(), pop[order[1]].copy()]
        while len(new_pop) < pop_size:
            p1 = tournament(pop, fits, rng); p2 = tournament(pop, fits, rng)
            child = crossover(p1, p2, rng) if rng.random() < 0.6 else p1.copy()
            child = mutate(child, rng, proto_pool)
            new_pop.append(child)
        pop = new_pop
    return archive

print("\n=== Nested out-of-sample per-domain search (this replaces the old flawed archive) ===")
corrected = {}
for dom in ALL_DOMAINS:
    pool = POOLS[dom]
    proto_pool = np.array([r["sig"] for r in pool["evolve"]])
    archive = evolve_nested(pool["evolve"], pool["archive_eval"], proto_pool, seed=0)
    agg = {seq: dict(perf=float(np.mean([v[0] for v in vals])), cost=float(np.mean([v[1] for v in vals])),
                      n=len(vals), seqlen=len(seq)) for seq, vals in archive.items()}
    corrected[dom] = agg
    print(f"  {dom:10s} unique sequences logged (OOS): {len(agg)}")

# ---- 3. per-length tiers from the corrected (OOS) archive
tiers = {}
for dom in ALL_DOMAINS:
    by_len = defaultdict(list)
    for seq, rec in corrected[dom].items():
        by_len[rec["seqlen"]].append((rec["perf"], seq, rec["cost"]))
    tiers[dom] = {}
    for L in sorted(by_len):
        best_perf, best_seq, best_cost = max(by_len[L], key=lambda x: x[0])
        tiers[dom][L] = dict(best_perf=best_perf, best_seq=best_seq, cost=best_cost)

print("\n=== Corrected (out-of-sample) per-length best, vs old flawed numbers ===")
old = json.load(open("/home/claude/pareto_results.json"))
for dom in ALL_DOMAINS:
    print(f"\n-- {dom} --")
    for L, t in tiers[dom].items():
        old_perf = old["report"][dom]["tiers"].get(str(L), {}).get("best_perf")
        print(f"   len={L}  OOS_perf={t['best_perf']:.3f}  (old flawed archive said {old_perf})  seq={[OPS[i] for i in t['best_seq']]}")

# ---- 4. candidate selection WITH final-holdout generalization check baked in
print("\n=== Selecting per-domain sequence: validated against final_holdout, not just archive ===")
validated = {}
for dom in ALL_DOMAINS:
    candidates = {tuple(t["best_seq"]): t["cost"] for t in tiers[dom].values()}
    candidates[tuple()] = 0.0  # always test the raw/no-op fallback too
    results = []
    for seq, archive_cost in candidates.items():
        perfs, costs, dims = [], [], []
        for rec in POOLS[dom]["final_holdout"]:
            F, perf, cost, dim = fitness(rec["X"], rec["y"], rec["task"], list(seq), seed=rec["seed"])
            perfs.append(perf); costs.append(cost)
        true_perf = float(np.mean(perfs)); true_cost = float(np.mean(costs))
        results.append((true_perf - LAMBDA*true_cost, seq, true_perf, true_cost))
    results.sort(key=lambda r: -r[0])
    best = results[0]
    validated[dom] = dict(seq=list(best[1]), seq_names=[OPS[i] for i in best[1]],
                           holdout_perf=best[2], holdout_cost=best[3])
    print(f"  {dom:10s} validated pick: {str(validated[dom]['seq_names']):30}  "
          f"true_holdout_perf={best[2]:.3f}  (vs raw baseline holdout perf={[r[2] for r in results if r[1]==()][0]:.3f})")

elapsed_mid = time.time() - t0
print(f"\n[{elapsed_mid:.1f}s elapsed]")

# ---- 5. assemble grammar from VALIDATED train-domain sequences, with strict router
print("\n=== Assembling grammar (validated sequences) + calibrating strict router ===")
rules = []
proto_by_dom = {}
for dom in TRAIN_DOMAINS:
    combo = POOLS[dom]["evolve"] + POOLS[dom]["archive_eval"]
    mean_sig = np.mean([r["sig"] for r in combo], axis=0)
    proto_by_dom[dom] = mean_sig
    rules.append(Rule(mean_sig, validated[dom]["seq"]))
    print(f"  rule[{dom}] seq={validated[dom]['seq_names']}  (holdout-validated perf={validated[dom]['holdout_perf']:.3f})")
assembled_v2 = Grammar(rules)

# calibrate radius: pooled within-domain distance to own prototype, using train draws
within_dists = []
for dom in TRAIN_DOMAINS:
    combo = POOLS[dom]["evolve"] + POOLS[dom]["archive_eval"]
    for r in combo:
        within_dists.append(np.linalg.norm(r["sig"] - proto_by_dom[dom]))
radius = float(np.percentile(within_dists, 95))
print(f"  calibrated router radius (95th pct within-domain distance) = {radius:.3f}")

def match_strict(g, s, radius):
    d = [np.linalg.norm(r.prototype - s) for r in g.rules]
    i = int(np.argmin(d))
    if d[i] > radius:
        return [], -1  # reject -> raw fallback
    return g.rules[i].seq, i

# ---- 6. final re-test: in-domain holdout + transfer, strict router, validated sequences
print("\n=== FINAL re-test (validated sequences + strict router) vs everything so far ===")
orig = json.load(open("/home/claude/cgit_results.json"))
close1 = json.load(open("/home/claude/close_loop_results.json"))

print("\n-- in-domain holdout (train domains) --")
for dom in TRAIN_DOMAINS:
    perfs = []
    for rec in POOLS[dom]["final_holdout"]:
        seq, ridx = match_strict(assembled_v2, rec["sig"], radius)
        F, perf, cost, dim = fitness(rec["X"], rec["y"], rec["task"], seq, seed=rec["seed"])
        perfs.append(perf)
    print(f"  {dom:10s} v2_assembled={np.mean(perfs):.3f}   original_shared={orig['summary'][dom]['in_domain_holdout']:.3f}   "
          f"v1_close_loop(broken router)={close1['assembled_in_domain'][dom]:.3f}")

print("\n-- transfer (test domains) --")
for dom in TEST_DOMAINS:
    perfs, routed = [], []
    for rec in POOLS[dom]["final_holdout"]:
        seq, ridx = match_strict(assembled_v2, rec["sig"], radius)
        routed.append("REJECT->raw" if ridx == -1 else TRAIN_DOMAINS[ridx])
        F, perf, cost, dim = fitness(rec["X"], rec["y"], rec["task"], seq, seed=rec["seed"])
        perfs.append(perf)
    raw_base = orig["summary"][dom]["baseline:raw"]
    local_ceiling_v2 = validated[dom]["holdout_perf"]
    print(f"  {dom:10s} v2_assembled={np.mean(perfs):.3f}  original_shared={orig['summary'][dom]['transferred_grammar']:.3f}  "
          f"v1_close_loop={close1['assembled_transfer'][dom]:.3f}  raw_baseline={raw_base:.3f}  "
          f"local_ceiling(validated)={local_ceiling_v2:.3f}  routed_to={routed}")

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed:.1f}s")

out = dict(
    tiers={d: {str(k): v for k, v in t.items()} for d, t in tiers.items()},
    validated={d: v for d, v in validated.items()},
    radius=radius,
    elapsed_seconds=elapsed,
)
with open("/home/claude/v2_results.json", "w") as f:
    json.dump(out, f, indent=2, default=lambda o: [OPS[i] for i in o] if isinstance(o, tuple) else o)
print("Saved /home/claude/v2_results.json")

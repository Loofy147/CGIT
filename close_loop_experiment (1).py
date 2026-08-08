import numpy as np, json, time
from cgit_lib import (
    DOMAIN_GENERATORS, TRAIN_DOMAINS, TEST_DOMAINS, OPS, OP_IDX,
    situation_signature, fitness, Rule, Grammar,
)

t0 = time.time()

def build_pool(domain, seeds):
    pool = []
    for s in seeds:
        X, y, task = DOMAIN_GENERATORS[domain](s)
        sig = situation_signature(X, y, task)
        pool.append(dict(X=X, y=y, task=task, sig=sig, seed=s))
    return pool

pareto = json.load(open("/home/claude/pareto_results.json"))["pareto"]

# ---- Step 4: cost-prune each domain's already-searched Pareto frontier ----
LAMBDA = 1.0  # same weight as the original F(G) = perf - cost - 0.5*fragility
chosen = {}
for dom in TRAIN_DOMAINS + TEST_DOMAINS:
    frontier = pareto[dom]  # list of [cost, perf, seq_names]
    best = max(frontier, key=lambda p: p[1] - LAMBDA * p[0])
    chosen[dom] = dict(cost=best[0], perf_local=best[1], seq=[OP_IDX[o] for o in best[2]], seq_names=best[2])
    print(f"{dom:10s} cost-pruned pick: {best[2]}  (local perf={best[1]:.3f}, cost={best[0]:.3f})")

# ---- Step 1-3 were already done in pareto_experiment.py (per-domain search + archive + frontier) ----
# ---- assemble: one rule per TRAIN domain, prototype = that domain's mean situation signature ----
print("\n=== Assembling grammar: one rule per train domain, Pareto-pruned sequence ===")
TRAIN_POOL_FOR_SIG = {d: build_pool(d, range(100, 105)) for d in TRAIN_DOMAINS}
rules = []
for dom in TRAIN_DOMAINS:
    mean_sig = np.mean([r["sig"] for r in TRAIN_POOL_FOR_SIG[dom]], axis=0)
    rules.append(Rule(mean_sig, chosen[dom]["seq"]))
    print(f"  rule[{dom}] proto={np.round(mean_sig,2)}  seq={chosen[dom]['seq_names']}")
assembled = Grammar(rules)

# ---- evaluate: identical protocol to the original transfer test ----
TRAIN_HOLDOUT = {d: build_pool(d, range(200, 202)) for d in TRAIN_DOMAINS}
EVAL_POOL = {d: build_pool(d, range(500, 504)) for d in TEST_DOMAINS}

print("\n=== In-domain holdout: assembled grammar vs original shared-evolution grammar ===")
orig = json.load(open("/home/claude/cgit_results.json"))
in_domain_orig = {d: orig["summary"].get(d, {}).get("in_domain_holdout") for d in TRAIN_DOMAINS}

assembled_in_domain = {}
for dom, pool in TRAIN_HOLDOUT.items():
    perfs = []
    for rec in pool:
        seq, ridx = assembled.match(rec["sig"])
        F, perf, cost, dim = fitness(rec["X"], rec["y"], rec["task"], seq, seed=rec["seed"])
        perfs.append(perf)
    assembled_in_domain[dom] = float(np.mean(perfs))
    print(f"  {dom:10s} assembled={assembled_in_domain[dom]:.3f}   original_shared_grammar={in_domain_orig[dom]}")

print("\n=== Transfer test: assembled grammar vs original vs raw baseline vs local ceiling ===")
routing = {}
assembled_transfer = {}
for dom, pool in EVAL_POOL.items():
    perfs, routed_to = [], []
    for rec in pool:
        seq, ridx = assembled.match(rec["sig"])
        routed_to.append(TRAIN_DOMAINS[ridx])
        F, perf, cost, dim = fitness(rec["X"], rec["y"], rec["task"], seq, seed=rec["seed"])
        perfs.append(perf)
    assembled_transfer[dom] = float(np.mean(perfs))
    routing[dom] = routed_to
    orig_transferred = orig["summary"][dom]["transferred_grammar"]
    orig_raw = orig["summary"][dom]["baseline:raw"]
    local_ceiling = max(p[1] for p in pareto[dom])
    print(f"  {dom:10s} assembled={assembled_transfer[dom]:.3f}  "
          f"original_shared={orig_transferred:.3f}  raw_baseline={orig_raw:.3f}  local_ceiling={local_ceiling:.3f}  "
          f"routed_to={routed_to}")

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed:.1f}s")

out = dict(chosen={d: {"seq": c["seq_names"], "cost": c["cost"], "local_perf": c["perf_local"]} for d, c in chosen.items()},
           assembled_in_domain=assembled_in_domain, in_domain_orig=in_domain_orig,
           assembled_transfer=assembled_transfer, routing=routing, elapsed_seconds=elapsed)
with open("/home/claude/close_loop_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved /home/claude/close_loop_results.json")

import numpy as np, json, time
from collections import defaultdict
from cgit_lib import (
    DOMAIN_GENERATORS, TRAIN_DOMAINS, TEST_DOMAINS, OPS, OP_IDX,
    situation_signature, fitness, fitness_auc, run_program,
    evaluate_representation, evaluate_representation_auc,
    Rule, Grammar, seeded_initial_population, mutate, crossover, tournament,
)

t0 = time.time()

def build_pool(domain, seeds):
    pool = []
    for s in seeds:
        X, y, task = DOMAIN_GENERATORS[domain](s)
        sig = situation_signature(X, y, task)
        pool.append(dict(X=X, y=y, task=task, sig=sig, seed=s))
    return pool

TRAIN_POOL = {d: build_pool(d, range(100, 105)) for d in TRAIN_DOMAINS}
TRAIN_HOLDOUT = {d: build_pool(d, range(200, 202)) for d in TRAIN_DOMAINS}
SW_TRAIN, SW_HOLDOUT = TRAIN_POOL["software"], TRAIN_HOLDOUT["software"]

# ============================================================ STEP 1 =====
# Discovered sequences + per-length Pareto frontier for software, AUC-scored
print("=== STEP 1: software-only performance-only search, AUC-scored ===")

def perf_only_fitness_logged_auc(g, pool, archive):
    total = 0.0
    for rec in pool:
        seq, _ = g.match(rec["sig"])
        F, perf, cost, dim = fitness_auc(rec["X"], rec["y"], rec["task"], seq, lambda_cost=0.0, seed=rec["seed"])
        archive[tuple(seq)].append((perf, cost))
        total += F
    return total / len(pool)

def evolve_perf_only_auc(pool, proto_pool, pop_size=20, generations=16, seed=0):
    rng = np.random.default_rng(seed)
    archive = defaultdict(list)
    pop = seeded_initial_population(pop_size, rng, proto_pool)
    best_g, best_f = None, -1e9
    for gen in range(generations):
        fits = [perf_only_fitness_logged_auc(g, pool, archive) for g in pop]
        gi = int(np.argmax(fits))
        if fits[gi] > best_f:
            best_f, best_g = fits[gi], pop[gi].copy()
        order = np.argsort(fits)[::-1]
        new_pop = [pop[order[0]].copy(), pop[order[1]].copy()]
        while len(new_pop) < pop_size:
            p1 = tournament(pop, fits, rng); p2 = tournament(pop, fits, rng)
            child = crossover(p1, p2, rng) if rng.random() < 0.6 else p1.copy()
            child = mutate(child, rng, proto_pool)
            new_pop.append(child)
        pop = new_pop
    return best_g, best_f, archive

proto_pool_sw = np.array([r["sig"] for r in SW_TRAIN])
_, _, archive_auc = evolve_perf_only_auc(SW_TRAIN, proto_pool_sw, seed=0)
agg_auc = {seq: dict(perf=float(np.mean([v[0] for v in vals])), cost=float(np.mean([v[1] for v in vals])),
                      n=len(vals), seqlen=len(seq)) for seq, vals in archive_auc.items()}
print(f"  unique sequences tried (AUC search): {len(agg_auc)}")

by_len_auc = defaultdict(list)
for seq, rec in agg_auc.items():
    by_len_auc[rec["seqlen"]].append((rec["perf"], seq, rec["cost"]))
tiers_auc = {}
for L in sorted(by_len_auc):
    best_perf, best_seq, best_cost = max(by_len_auc[L], key=lambda x: x[0])
    tiers_auc[L] = dict(best_perf=best_perf, best_seq=[OPS[i] for i in best_seq], cost=best_cost)
    print(f"   len={L}  best_perf(AUC)={best_perf:.3f}  seq={tiers_auc[L]['best_seq']}")

old = json.load(open("/home/claude/pareto_results.json"))
old_tiers = old["report"]["software"]["tiers"]
print("\n  -- old (accuracy-clipped) tiers for comparison --")
for L, t in old_tiers.items():
    print(f"   len={L}  best_perf(acc)={t['best_perf']:.3f}  seq={t['best_seq']}")

def pareto_frontier(points):
    pts = sorted(points, key=lambda p: p[0]); frontier = []; best_perf = -1
    for cost, perf, seq in pts:
        if perf > best_perf:
            frontier.append((cost, perf, seq)); best_perf = perf
    return frontier

pts_auc = [(rec["cost"], rec["perf"], seq) for seq, rec in agg_auc.items()]
frontier_auc = [(c, p, [OPS[i] for i in s]) for c, p, s in pareto_frontier(pts_auc)]
print("\n  AUC Pareto frontier:")
for c, p, s in frontier_auc:
    print(f"     cost={c:.3f}  perf={p:.3f}  seq={s}")

# ============================================================ STEP 1b ====
# Critical check: does the AUC-favored deep sequence actually generalize to a
# FRESH software holdout, or is it overfit to the 5-draw search pool (as the
# accuracy-based COMPRESS->PREDICT->RELATE->ABSTRACT pick turned out to be)?
print("\n=== STEP 1b: holdout generalization check (fresh unseen software draws) ===")
candidates = {f"len{L}_auc_best": t["best_seq"] for L, t in tiers_auc.items()}
candidates["len4_accuracy_best(from earlier run)"] = old_tiers["4"]["best_seq"] if "4" in old_tiers else old_tiers[list(old_tiers)[-1]]["best_seq"]
gen_report = {}
for name, seq_names in candidates.items():
    seq = [OP_IDX[o] for o in seq_names]
    accs, aucs = [], []
    for rec in SW_HOLDOUT:
        R = run_program(rec["X"], seq)
        accs.append(evaluate_representation(R, rec["y"], rec["task"], seed=rec["seed"])["perf"])
        ev = evaluate_representation_auc(R, rec["y"], rec["task"], seed=rec["seed"])
        aucs.append(ev["perf"])
    gen_report[name] = dict(seq=seq_names, holdout_acc_perf=float(np.mean(accs)), holdout_auc_perf=float(np.mean(aucs)))
    print(f"  {name:35s} seq={seq_names}")
    print(f"       holdout accuracy-based perf={np.mean(accs):.3f}   holdout AUC-based perf={np.mean(aucs):.3f}")

# ============================================================ STEP 2 =====
# Rerun the ORIGINAL joint 3-domain grammar evolution, but software's fitness
# contribution is now AUC-scored (physics/biology unchanged). This directly
# tests whether the joint search's software rule improves once its gradient
# isn't flattened relative to physics/biology's much larger, ungapped signal.
print("\n=== STEP 2: joint 3-domain evolution, software=AUC / physics+biology=original ===")

def mixed_grammar_fitness(g, pool_by_domain):
    total, n = 0.0, 0
    for dom, pool in pool_by_domain.items():
        fn = fitness_auc if dom == "software" else fitness
        for rec in pool:
            seq, _ = g.match(rec["sig"])
            F, perf, cost, dim = fn(rec["X"], rec["y"], rec["task"], seq, seed=rec["seed"])
            total += F; n += 1
    return total / n

proto_pool_train = np.array([r["sig"] for pool in TRAIN_POOL.values() for r in pool])

def evolve_mixed(pool_by_domain, proto_pool, pop_size=18, generations=18, seed=0):
    rng = np.random.default_rng(seed)
    pop = seeded_initial_population(pop_size, rng, proto_pool)
    best_g, best_f = None, -1e9
    for gen in range(generations):
        fits = [mixed_grammar_fitness(g, pool_by_domain) for g in pop]
        gi = int(np.argmax(fits))
        if fits[gi] > best_f:
            best_f, best_g = fits[gi], pop[gi].copy()
        order = np.argsort(fits)[::-1]
        new_pop = [pop[order[0]].copy(), pop[order[1]].copy()]
        while len(new_pop) < pop_size:
            p1 = tournament(pop, fits, rng); p2 = tournament(pop, fits, rng)
            child = crossover(p1, p2, rng) if rng.random() < 0.6 else p1.copy()
            child = mutate(child, rng, proto_pool)
            new_pop.append(child)
        pop = new_pop
    return best_g, best_f

mixed_g, mixed_f = evolve_mixed(TRAIN_POOL, proto_pool_train, seed=0)
print(f"  mixed-evolution best F={mixed_f:.4f}")

print("  in-domain holdout under the mixed-evolved grammar:")
for dom, pool in TRAIN_HOLDOUT.items():
    accs = []
    for rec in pool:
        seq, _ = mixed_g.match(rec["sig"])
        F, perf, cost, dim = fitness(rec["X"], rec["y"], rec["task"], seq, seed=rec["seed"])
        accs.append(perf)
    tag = "  <-- was 0.000 in the original joint run" if dom == "software" else ""
    print(f"    {dom:10s} accuracy-based perf={np.mean(accs):.3f}{tag}")

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed:.1f}s")

out = dict(tiers_auc=tiers_auc, old_tiers_accuracy=old_tiers, frontier_auc=frontier_auc,
           holdout_generalization=gen_report, mixed_evolution_fitness=mixed_f,
           elapsed_seconds=elapsed)
with open("/home/claude/auc_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved /home/claude/auc_results.json")

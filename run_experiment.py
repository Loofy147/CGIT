"""
Cognitive Grammar Induction Test (CGIT) — Run Evolutionary Experiment.
Trains a Cognitive Grammar on a split of source domains, and validates
generalization and transfer capability on unseen target domains.
"""

import numpy as np
import cgit_lib

def run():
    # 1. Setup seed and reproducibility
    seed = 42
    rng = np.random.default_rng(seed)
    print("=" * 70)
    print("CGIT Evolutionary Grammar Induction System — Running Experiment")
    print("=" * 70)

    # 2. Build prototype pool from training domains
    print("\n[Phase 1] Building Prototype Pool from Train Domains...")
    proto_pool = []
    # Collect sit signatures from TRAIN_DOMAINS ('physics', 'software', 'biology') across seeds 100 to 109
    for domain in cgit_lib.TRAIN_DOMAINS:
        for seed_val in range(100, 110):
            X, y, task = cgit_lib.DOMAIN_GENERATORS[domain](seed_val)
            sig = cgit_lib.situation_signature(X, y, task)
            proto_pool.append(sig)

    print(f"Prototype pool size: {len(proto_pool)} sit signatures.")
    print("Example signatures (first 3):")
    for i, sig in enumerate(proto_pool[:3]):
        print(f"  Sig {i}: {sig}")

    # 3. Initialize Population
    pop_size = 40
    generations = 30
    print(f"\n[Phase 2] Initializing GA Population of size {pop_size}...")
    pop = cgit_lib.seeded_initial_population(pop_size, rng, proto_pool)

    # Helper function to evaluate fitness of a Grammar on training split
    # Training split: 3 train domains x 5 seeds (0 to 4) = 15 training conditions
    def eval_grammar_train(g):
        fit_vals = []
        for d_name in cgit_lib.TRAIN_DOMAINS:
            for s_val in range(5):
                X, y, task = cgit_lib.DOMAIN_GENERATORS[d_name](s_val)
                sig = cgit_lib.situation_signature(X, y, task)
                op_seq, rule_idx = g.match(sig)
                f_val, perf, cost, out_dim = cgit_lib.fitness(
                    X, y, task, op_seq, lambda_cost=1.0, seed=s_val, cheap_fragility=True
                )
                fit_vals.append(f_val)
        return np.mean(fit_vals)

    # 4. Evolutionary Loop
    print(f"\n[Phase 3] Running Evolutionary Search for {generations} Generations...")
    best_grammar = None
    best_fitness_history = []
    mean_fitness_history = []

    for gen in range(generations):
        # Calculate fitness for all individuals
        fits = [eval_grammar_train(g) for g in pop]

        # Track best
        best_idx = np.argmax(fits)
        best_fit = fits[best_idx]
        mean_fit = np.mean(fits)

        best_grammar = pop[best_idx].copy()
        best_fitness_history.append(best_fit)
        mean_fitness_history.append(mean_fit)

        print(f"Gen {gen:02d}/{generations:02d} | Best Fitness: {best_fit:.5f} | Mean Fitness: {mean_fit:.5f} | Rules: {len(best_grammar.rules)}")

        # Create next population
        new_pop = []
        # Elitism: keep the best grammar unchanged
        new_pop.append(best_grammar.copy())

        while len(new_pop) < pop_size:
            # Tournament selection for parents
            if rng.random() < 0.8:
                # Crossover + Mutation
                p1 = cgit_lib.tournament(pop, fits, rng, k=3)
                p2 = cgit_lib.tournament(pop, fits, rng, k=3)
                child = cgit_lib.crossover(p1, p2, rng)
                child_mut = cgit_lib.mutate(child, rng, proto_pool)
                new_pop.append(child_mut)
            else:
                # Mutation only
                p = cgit_lib.tournament(pop, fits, rng, k=3)
                child_mut = cgit_lib.mutate(p, rng, proto_pool)
                new_pop.append(child_mut)

        pop = new_pop

    print("\nEvolution complete!")
    print(f"Best Evolved Grammar Training Fitness: {best_fitness_history[-1]:.5f}")
    print("Best Evolved Grammar Rules:")
    for idx, rule in enumerate(best_grammar.rules):
        seq_names = [cgit_lib.OPS[op_idx] for op_idx in rule.seq]
        print(f"  Rule {idx}: Prototype {rule.prototype} -> {seq_names}")

    # 5. Define Baseline Grammars for Evaluation
    # G1: Always COMPRESS -> PREDICT
    g1_baseline = cgit_lib.Grammar([cgit_lib.Rule(np.zeros(cgit_lib.SIT_DIM), [cgit_lib.OP_IDX["COMPRESS"], cgit_lib.OP_IDX["PREDICT"]])])
    # G2: Always DISTINGUISH -> RELATE
    g2_baseline = cgit_lib.Grammar([cgit_lib.Rule(np.zeros(cgit_lib.SIT_DIM), [cgit_lib.OP_IDX["DISTINGUISH"], cgit_lib.OP_IDX["RELATE"]])])

    # 6. Empirical Validation on Unseen Test Domains
    # Split: TEST_DOMAINS ('economics', 'robotics', 'social') across seeds 200 to 209 (30 test conditions total)
    print("\n[Phase 4] Evaluating Evolved Grammar and Baselines on Unseen Test Domains...")

    # Track statistics per domain and per grammar
    # Grammars under test: Evolved, G1 Baseline, G2 Baseline, G3 Baseline (stochastic random sequence per seed)
    results = {
        "Evolved": {d: [] for d in cgit_lib.TEST_DOMAINS},
        "G1_Compress_Predict": {d: [] for d in cgit_lib.TEST_DOMAINS},
        "G2_Distinguish_Relate": {d: [] for d in cgit_lib.TEST_DOMAINS},
        "G3_Random": {d: [] for d in cgit_lib.TEST_DOMAINS}
    }

    test_seeds = list(range(200, 210))
    for d_name in cgit_lib.TEST_DOMAINS:
        for seed_val in test_seeds:
            # Generate target dataset
            X, y, task = cgit_lib.DOMAIN_GENERATORS[d_name](seed_val)
            sig = cgit_lib.situation_signature(X, y, task)

            # --- Evaluate Evolved Grammar ---
            op_seq_evolved, _ = best_grammar.match(sig)
            fit_evolved, perf_evolved, cost_evolved, dim_evolved = cgit_lib.fitness(
                X, y, task, op_seq_evolved, lambda_cost=1.0, seed=seed_val, cheap_fragility=True
            )
            frag_res = cgit_lib.evaluate_representation(
                cgit_lib.run_program(X, op_seq_evolved), y, task, seed=seed_val,
                want_true_fragility=True, X_for_noise=X, op_seq=op_seq_evolved
            )
            true_frag_evolved = frag_res["fragility"]
            results["Evolved"][d_name].append({
                "fitness": fit_evolved, "perf": perf_evolved, "cost": cost_evolved, "dim": dim_evolved, "fragility": true_frag_evolved, "seq": op_seq_evolved
            })

            # --- Evaluate G1 Baseline ---
            op_seq_g1, _ = g1_baseline.match(sig)
            fit_g1, perf_g1, cost_g1, dim_g1 = cgit_lib.fitness(
                X, y, task, op_seq_g1, lambda_cost=1.0, seed=seed_val, cheap_fragility=True
            )
            frag_res_g1 = cgit_lib.evaluate_representation(
                cgit_lib.run_program(X, op_seq_g1), y, task, seed=seed_val,
                want_true_fragility=True, X_for_noise=X, op_seq=op_seq_g1
            )
            true_frag_g1 = frag_res_g1["fragility"]
            results["G1_Compress_Predict"][d_name].append({
                "fitness": fit_g1, "perf": perf_g1, "cost": cost_g1, "dim": dim_g1, "fragility": true_frag_g1, "seq": op_seq_g1
            })

            # --- Evaluate G2 Baseline ---
            op_seq_g2, _ = g2_baseline.match(sig)
            fit_g2, perf_g2, cost_g2, dim_g2 = cgit_lib.fitness(
                X, y, task, op_seq_g2, lambda_cost=1.0, seed=seed_val, cheap_fragility=True
            )
            frag_res_g2 = cgit_lib.evaluate_representation(
                cgit_lib.run_program(X, op_seq_g2), y, task, seed=seed_val,
                want_true_fragility=True, X_for_noise=X, op_seq=op_seq_g2
            )
            true_frag_g2 = frag_res_g2["fragility"]
            results["G2_Distinguish_Relate"][d_name].append({
                "fitness": fit_g2, "perf": perf_g2, "cost": cost_g2, "dim": dim_g2, "fragility": true_frag_g2, "seq": op_seq_g2
            })

            # --- Evaluate G3 Stochastic Baseline (Fresh random sequence per test seed) ---
            op_seq_g3 = cgit_lib.random_seq(rng, minlen=1, maxlen=4)
            fit_g3, perf_g3, cost_g3, dim_g3 = cgit_lib.fitness(
                X, y, task, op_seq_g3, lambda_cost=1.0, seed=seed_val, cheap_fragility=True
            )
            frag_res_g3 = cgit_lib.evaluate_representation(
                cgit_lib.run_program(X, op_seq_g3), y, task, seed=seed_val,
                want_true_fragility=True, X_for_noise=X, op_seq=op_seq_g3
            )
            true_frag_g3 = frag_res_g3["fragility"]
            results["G3_Random"][d_name].append({
                "fitness": fit_g3, "perf": perf_g3, "cost": cost_g3, "dim": dim_g3, "fragility": true_frag_g3, "seq": op_seq_g3
            })

    # 7. Print and Collate Summary Reports
    print("\n" + "=" * 70)
    print("CGIT EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 70)

    # Store aggregated metrics for report generation
    report_data = []

    for d_name in cgit_lib.TEST_DOMAINS:
        print(f"\nTarget Domain: {d_name.upper()}")
        print("-" * 50)
        for g_name in results.keys():
            fits = [r["fitness"] for r in results[g_name][d_name]]
            perfs = [r["perf"] for r in results[g_name][d_name]]
            costs = [r["cost"] for r in results[g_name][d_name]]
            dims = [r["dim"] for r in results[g_name][d_name]]
            frags = [r["fragility"] for r in results[g_name][d_name]]

            mean_fit, std_fit = np.mean(fits), np.std(fits)
            mean_perf, std_perf = np.mean(perfs), np.std(perfs)
            mean_cost, std_cost = np.mean(costs), np.std(costs)
            mean_dim, std_dim = np.mean(dims), np.std(dims)
            mean_frag, std_frag = np.mean(frags), np.std(frags)

            print(f"Grammar: {g_name:<25}")
            print(f"  Fitness:        {mean_fit:.5f} ± {std_fit:.5f}")
            print(f"  Performance:    {mean_perf:.5f} ± {std_perf:.5f}")
            print(f"  Cost:           {mean_cost:.5f} ± {std_cost:.5f}")
            print(f"  Out Dimension:  {mean_dim:.1f} ± {std_dim:.1f}")
            print(f"  True Fragility: {mean_frag:.5f} ± {std_frag:.5f}")
            print()

            report_data.append({
                "domain": d_name,
                "grammar": g_name,
                "mean_fit": mean_fit, "std_fit": std_fit,
                "mean_perf": mean_perf, "std_perf": std_perf,
                "mean_cost": mean_cost, "std_cost": std_cost,
                "mean_dim": mean_dim, "std_dim": std_dim,
                "mean_frag": mean_frag, "std_frag": std_frag
            })

    # Save to a format easily readable by other processes
    return best_grammar, report_data

if __name__ == "__main__":
    run()

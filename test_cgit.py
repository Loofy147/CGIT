import unittest
import numpy as np
import cgit_lib
import run_experiment

class TestCGITSystem(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        # Generate dummy data for testing
        self.X = self.rng.normal(0, 1, (50, 10))
        self.y_reg = self.rng.normal(0, 1, 50)
        self.y_clf = (self.rng.uniform(0, 1, 50) > 0.5).astype(int)

    def test_situation_signature(self):
        # Test signature for regression
        sig_reg = cgit_lib.situation_signature(self.X, self.y_reg, "reg")
        self.assertEqual(len(sig_reg), cgit_lib.SIT_DIM)
        self.assertFalse(np.isnan(sig_reg).any())
        self.assertFalse(np.isinf(sig_reg).any())

        # Test signature for classification
        sig_clf = cgit_lib.situation_signature(self.X, self.y_clf, "clf")
        self.assertEqual(len(sig_clf), cgit_lib.SIT_DIM)

    def test_interact_operator(self):
        R = np.array([[2.0, 3.0, 5.0],
                      [4.0, -1.0, 0.0]])
        res = cgit_lib.op_interact(R)
        expected = np.array([[2.0, 3.0, 5.0, 6.0, 10.0, 15.0],
                             [4.0, -1.0, 0.0, -4.0, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(res, expected)

    def test_primitive_operators(self):
        # Test each of the 8 primitive operators to ensure dimension constraints
        for op_idx, op_func in enumerate(cgit_lib.OP_FUNCS):
            R = op_func(self.X)
            # All operator outputs must be 2D arrays with the same row count
            self.assertEqual(len(R.shape), 2)
            self.assertEqual(R.shape[0], self.X.shape[0])
            self.assertFalse(np.isnan(R).any())

    def test_run_program(self):
        # Test run_program with multiple operators
        op_seq = [cgit_lib.OP_IDX["COMPRESS"], cgit_lib.OP_IDX["PREDICT"]]
        R = cgit_lib.run_program(self.X, op_seq)
        self.assertEqual(len(R.shape), 2)
        self.assertEqual(R.shape[0], self.X.shape[0])
        # Ensure dimensions don't exceed MAX_DIM
        self.assertTrue(R.shape[1] <= cgit_lib.MAX_DIM)

    def test_grammar_matching(self):
        # Create a simple grammar
        proto = np.zeros(cgit_lib.SIT_DIM)
        rule1 = cgit_lib.Rule(proto, [cgit_lib.OP_IDX["COMPRESS"]])
        grammar = cgit_lib.Grammar([rule1])

        # Match a dummy signature
        sig = np.ones(cgit_lib.SIT_DIM)
        seq, matched_idx = grammar.match(sig)
        self.assertEqual(seq, [cgit_lib.OP_IDX["COMPRESS"]])
        self.assertEqual(matched_idx, 0)

    def test_seeded_initial_population(self):
        proto_pool = [np.zeros(cgit_lib.SIT_DIM), np.ones(cgit_lib.SIT_DIM)]
        pop = cgit_lib.seeded_initial_population(10, self.rng, proto_pool)
        self.assertEqual(len(pop), 10)
        # Verify first pop elements are G1, G2, G3 as defined
        self.assertEqual(len(pop[0].rules), 1)
        self.assertEqual(pop[0].rules[0].seq, [cgit_lib.OP_IDX["COMPRESS"], cgit_lib.OP_IDX["PREDICT"]])

if __name__ == "__main__":
    unittest.main()

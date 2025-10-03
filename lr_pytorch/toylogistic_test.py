import unittest
from toylogistic_buzzer import ToyLogisticBuzzer, Example

# ---------------------------------------------------------------------
# Helper: print feature ↔ weight mapping
# ---------------------------------------------------------------------
def show_beta(feature_names, beta_vector, title="β weights"):
    mapping = {fname: round(beta_vector[i], 6) for i, fname in enumerate(feature_names)}
    print(f"\n{title}:")
    for k, v in mapping.items():
        print(f"  {k:<12} {v:+.6f}")
    return mapping               # handy if you want to inspect programmatically


kTOY_VOCAB = "BIAS_CONSTANT A B C D".split()
kPOS = Example({"label": True,  "A": 4, "B": 3, "C": 1}, kTOY_VOCAB)
kNEG = Example({"label": False, "B": 1, "C": 3, "D": 4}, kTOY_VOCAB)

class TestLogReg(unittest.TestCase):
    def setUp(self):
        self.logreg_unreg = ToyLogisticBuzzer(num_features=5, learning_rate=1.0)

    def test_unreg(self):
        # -------- first update (positive example) ---------------------
        print("\nBefore any update:")
        show_beta(kTOY_VOCAB, self.logreg_unreg._beta)

        print("\nkPOS.x:", kPOS.x)
        beta = self.logreg_unreg.sg_update(kPOS, 0)          # SGD step 1
        show_beta(kTOY_VOCAB, beta, title="After first update")

        # exact checks
        self.assertAlmostEqual(beta[0], 0.5)
        self.assertAlmostEqual(beta[1], 2.0)
        self.assertAlmostEqual(beta[2], 1.5)
        self.assertAlmostEqual(beta[3], 0.5)
        self.assertAlmostEqual(beta[4], 0.0)

        # -------- second update (negative example) -------------------
        print("\nkNEG.x:", kNEG.x)
        beta = self.logreg_unreg.sg_update(kNEG, 1)          # SGD step 2
        show_beta(kTOY_VOCAB, beta, title="After second update")

        # exact checks
        self.assertAlmostEqual(beta[0], -0.47068776924864364)
        self.assertAlmostEqual(beta[1],  2.0)
        self.assertAlmostEqual(beta[2],  0.5293122307513564)
        self.assertAlmostEqual(beta[3], -2.4120633077459308)
        self.assertAlmostEqual(beta[4], -3.8827510769945746)

if __name__ == "__main__":
    unittest.main()

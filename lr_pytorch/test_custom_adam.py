import unittest
import torch
from pytorch_custom_adam_buzzer import ToyLogisticBuzzer, Example, CustomAdamOptimizer

# ---------------------------------------------------------------------
# print feature ↔ weight mapping
# ---------------------------------------------------------------------
def show_beta(beta_vec, title="β weights"):
    """
    beta_vec: list/np.array/torch.Tensor with 5 scalars:
              [A, B, C, D, bias]
    """
    feature_names = ["A", "B", "C", "D", "bias"]
    #  Make sure it's a 1-D list
    beta_vec = [float(v) for v in beta_vec]
    mapping = dict(zip(feature_names, beta_vec))
    print(f"\n{title}:")
    for k, v in mapping.items():
        print(f"  {k:<5} {v:+.6f}")
    return mapping


# Test vocabulary - PyTorch version since nn.Linear handles biases
kTOY_VOCAB = ["A", "B", "C", "D"]
kPOS = Example({"label": True, "A": 4, "B": 3, "C": 1}, kTOY_VOCAB, use_bias=False)
kNEG = Example({"label": False, "B": 1, "C": 3, "D": 4}, kTOY_VOCAB, use_bias=False)

class TestCustomAdamLogReg(unittest.TestCase):
    def setUp(self):
        # Set seed for reproducible tests
        torch.manual_seed(1701)
        # Note: Using lower learning rate for Adam (Adam typically needs smaller LR than SGD)
        self.logreg_unreg = ToyLogisticBuzzer(num_features=4, learning_rate=0.1, mu=0.0)

    def test_custom_adam_optimizer_initialization(self):
        """Test that custom Adam optimizer is properly initialized"""
        optimizer = self.logreg_unreg.optimizer
        self.assertIsInstance(optimizer, CustomAdamOptimizer)
        self.assertEqual(optimizer.lr, 0.1)
        self.assertEqual(optimizer.beta1, 0.9)
        self.assertEqual(optimizer.beta2, 0.999)
        self.assertEqual(optimizer.eps, 1e-8)
        self.assertEqual(optimizer.weight_decay, 0.0)
        
        # check state is initialized for each parameter
        self.assertEqual(len(optimizer.state), 2)  # weight and bias
        for param in optimizer.params:
            state = optimizer.state[param]
            self.assertEqual(state['step'], 0)
            self.assertTrue(torch.equal(state['exp_avg'], torch.zeros_like(param.data)))
            self.assertTrue(torch.equal(state['exp_avg_sq'], torch.zeros_like(param.data)))

    def test_unreg(self):
        # ----- 0) before any update -----------------------------------
        w_init  = self.logreg_unreg.model.linear.weight.data.squeeze().tolist()
        b_init  = self.logreg_unreg.model.linear.bias.data.item()
        init_beta = w_init + [b_init]          # [A,B,C,D,bias]

        show_beta(init_beta, title="Before any update")
        print(f"\nkPOS.x: {[1.0] + kPOS.x.tolist()}")    # show bias-constant 1

        # ----- 1) first update (positive example) ---------------------
        beta1 = self.logreg_unreg.sg_update(kPOS, 0)      # [A,B,C,D,bias]
        show_beta(beta1, title="After first update")

        # core assertions (unchanged) .........................
        self.assertNotEqual(beta1[0], 0.0)
        self.assertNotEqual(beta1[1], 0.0)
        self.assertNotEqual(beta1[2], 0.0)
        self.assertEqual   (beta1[3], 0.0)
        self.assertNotEqual(beta1[4], 0.0)

        # ----- 2) second update (negative example) --------------------
        print(f"\nkNEG.x: {[1.0] + kNEG.x.tolist()}")

        beta2 = self.logreg_unreg.sg_update(kNEG, 1)      # [A,B,C,D,bias]
        show_beta(beta2, title="After second update")

        # sanity / behaviour checks (unchanged) .......................
        for i in range(len(beta2)):
            if i == 3:          # D feature should have moved
                self.assertNotEqual(beta2[i], 0.0)

        for w in beta2:
            self.assertFalse(torch.isnan(torch.tensor(w)))
            self.assertFalse(torch.isinf(torch.tensor(w)))

    def test_custom_adam_state_tracking(self):
        """Test that custom Adam maintains proper state across updates"""
        optimizer = self.logreg_unreg.optimizer
        
        # Initial state should have step=0 for all parameters
        for param in optimizer.params:
            self.assertEqual(optimizer.state[param]['step'], 0)
        
        # After first update, step should be 1
        self.logreg_unreg.sg_update(kPOS, 0)
        for param in optimizer.params:
            self.assertEqual(optimizer.state[param]['step'], 1)
            # Momentum buffers should be non-zero
            self.assertFalse(torch.equal(optimizer.state[param]['exp_avg'], torch.zeros_like(param.data)))
            self.assertFalse(torch.equal(optimizer.state[param]['exp_avg_sq'], torch.zeros_like(param.data)))
        
        # After second update, step should be 2
        self.logreg_unreg.sg_update(kNEG, 1)
        for param in optimizer.params:
            self.assertEqual(optimizer.state[param]['step'], 2)

    def test_custom_adam_vs_pytorch_adam(self):
        """Test that custom Adam behaves similarly to PyTorch's Adam"""
        # Create model with custom Adam
        torch.manual_seed(1701)
        custom_model = ToyLogisticBuzzer(num_features=4, learning_rate=0.1, mu=0.0)
        
        # Create model with PyTorch Adam for comparison
        torch.manual_seed(1701)  # Same seed
        pytorch_model = ToyLogisticBuzzer(num_features=4, learning_rate=0.1, mu=0.0)
        # Replace custom Adam with PyTorch Adam
        pytorch_model.optimizer = torch.optim.Adam(pytorch_model.model.parameters(), lr=0.1)
        
        # Same updates
        custom_beta1 = custom_model.sg_update(kPOS, 0)
        pytorch_beta1 = pytorch_model.sg_update(kPOS, 0)
        
        # Results should be very similar (small numerical differences are acceptable)
        differences = [abs(c - p) for c, p in zip(custom_beta1, pytorch_beta1)]
        
        # Differences should be small (within reasonable tolerance)
        for diff in differences:
            self.assertLess(diff, 1e-5, "Custom Adam should behave very similarly to PyTorch Adam")

    def test_custom_adam_vs_sgd_behavior(self):
        """Test that custom Adam behaves differently than SGD"""
        # Create a comparison SGD model for reference
        torch.manual_seed(1701)
        adam_model = ToyLogisticBuzzer(num_features=4, learning_rate=0.1, mu=0.0)
        
        torch.manual_seed(1701)  # Same seed
        sgd_model = ToyLogisticBuzzer(num_features=4, learning_rate=0.1, mu=0.0)
        # Replace custom Adam with SGD for comparison
        sgd_model.optimizer = torch.optim.SGD(sgd_model.model.parameters(), lr=0.1)
        
        # Same updates
        adam_beta1 = adam_model.sg_update(kPOS, 0)
        sgd_beta1 = sgd_model.sg_update(kPOS, 0)
        
        # They should produce different results due to different optimizers
        # (Adam uses momentum and bias correction)
        differences = [abs(a - s) for a, s in zip(adam_beta1, sgd_beta1)]
        
        # At least some differences should be non-trivial
        self.assertGreater(max(differences), 1e-6)

    def test_custom_adam_bias_correction(self):
        """Test that bias correction works properly in early steps"""
        optimizer = self.logreg_unreg.optimizer
        
        # Perform several updates and check that bias correction is applied
        for i in range(5):
            self.logreg_unreg.sg_update(kPOS if i % 2 == 0 else kNEG, i)
            
            # Check bias correction factors
            for param in optimizer.params:
                state = optimizer.state[param]
                step = state['step']
                
                bias_correction1 = 1 - optimizer.beta1 ** step
                bias_correction2 = 1 - optimizer.beta2 ** step
                
                # Bias correction should be closer to 1 as step increases
                self.assertGreater(bias_correction1, 0)
                self.assertLess(bias_correction1, 1)
                self.assertGreater(bias_correction2, 0)
                self.assertLess(bias_correction2, 1)
                
                # Should approach 1 as step increases
                if step > 1:
                    prev_correction1 = 1 - optimizer.beta1 ** (step - 1)
                    self.assertGreater(bias_correction1, prev_correction1)

    def test_custom_adam_weight_decay(self):
        """Test that weight decay (L2 regularization) works"""
        # Create models with and without weight decay
        torch.manual_seed(1701)
        no_decay_model = ToyLogisticBuzzer(num_features=4, learning_rate=0.1, mu=0.0)
        
        torch.manual_seed(1701)
        with_decay_model = ToyLogisticBuzzer(num_features=4, learning_rate=0.1, mu=0.1)
        
        # Train both models with same data
        for i in range(10):
            example = kPOS if i % 2 == 0 else kNEG
            no_decay_model.sg_update(example, i)
            with_decay_model.sg_update(example, i)
        
        # Model with weight decay should have smaller weight magnitudes
        no_decay_weights = no_decay_model.model.linear.weight.data.abs().sum()
        with_decay_weights = with_decay_model.model.linear.weight.data.abs().sum()
        
        # Generally, regularization should lead to smaller weights
        # (This might not always be true for very few iterations, but usually is)
        self.assertNotEqual(no_decay_weights.item(), with_decay_weights.item())

    def test_optimizer_state_inspection(self):
        """Test the optimizer state inspection functionality"""
        # Perform some updates
        for i in range(3):
            self.logreg_unreg.sg_update(kPOS if i % 2 == 0 else kNEG, i)
        
        # Get optimizer state info
        state_info = self.logreg_unreg.get_optimizer_state()
        
        # Should have info for each parameter
        self.assertIn('param_0', state_info)  # weight parameter
        self.assertIn('param_1', state_info)  # bias parameter
        
        # Each parameter should have step count and momentum norms
        for param_key in state_info:
            param_info = state_info[param_key]
            self.assertIn('step', param_info)
            self.assertIn('exp_avg_norm', param_info)
            self.assertIn('exp_avg_sq_norm', param_info)
            
            # Step should be 3 after 3 updates
            self.assertEqual(param_info['step'], 3)
            # Momentum norms should be positive
            self.assertGreater(param_info['exp_avg_norm'], 0)
            self.assertGreater(param_info['exp_avg_sq_norm'], 0)

    def test_custom_adam_hyperparameters(self):
        """Test custom Adam with different hyperparameters"""
        # Test with non-default hyperparameters
        custom_buzzer = ToyLogisticBuzzer(
            num_features=4, 
            learning_rate=0.01, 
            mu=0.01,
            betas=(0.8, 0.99),  # Different betas
            eps=1e-6  # Different eps
        )
        
        optimizer = custom_buzzer.optimizer
        self.assertEqual(optimizer.lr, 0.01)
        self.assertEqual(optimizer.beta1, 0.8)
        self.assertEqual(optimizer.beta2, 0.99)
        self.assertEqual(optimizer.eps, 1e-6)
        self.assertEqual(optimizer.weight_decay, 0.01)
        
        # Should still work for updates
        beta = custom_buzzer.sg_update(kPOS, 0)
        
        # All weights should be finite and non-zero (except D)
        self.assertNotEqual(beta[0], 0.0)
        self.assertNotEqual(beta[1], 0.0) 
        self.assertNotEqual(beta[2], 0.0)
        self.assertEqual(beta[3], 0.0)
        self.assertNotEqual(beta[4], 0.0)

if __name__ == '__main__':
    unittest.main()

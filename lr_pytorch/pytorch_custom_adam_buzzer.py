import pickle
import random
from math import exp, log, sqrt
from collections import defaultdict
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import issparse
from buzzer import Buzzer

kSEED = 1701
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)
torch.manual_seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Note: Prevents overflow of exp by capping activation at 20.

    :param score: A real valued number to convert into a number between 0 and 1
    """
    if abs(score) > threshold:
        score = threshold * np.sign(score)

    activation = exp(score)
    return activation / (1.0 + activation)

def create_feature_matrix_sklearn(train_data, test_data=None):
    """
    Turn Example objects into NumPy feature matrices with sklearn's DictVectorizer.

    Returns
    -------
    X_train : np.ndarray
    y_train : np.ndarray
    X_test  : np.ndarray | None
    y_test  : np.ndarray | None
    vectorizer : DictVectorizer
    """

    # Get the Dictionary and Labels for training data
    train_dicts  = [ex.raw_features for ex in train_data]
    y_train      = np.array([ex.y for ex in train_data], dtype=np.float32)

    '''
    TODOs: create a data iterator for train and test data
    '''

    # -----------------------------------------------------------
    # Implement: Fit a vectorizer on training data into a  sparse matrix
    # -----------------------------------------------------------
    vectorizer   = None
    X_train = None

    # Implement: Convert to Dense for PyTorch

    # -----------------------------------------------------------
    # Add the same data processing for test data if it exists
    # -----------------------------------------------------------
    if test_data:
        X_test, y_test = None, None
    else:
        X_test, y_test = None, None

    return X_train, y_train, X_test, y_test, vectorizer


class SimpleLogreg(nn.Module):
    """
    Simple logistic regression model using PyTorch
    """
    def __init__(self, num_features):
        super(SimpleLogreg, self).__init__()

        '''
        TODOs: Implement a single linear layer with the number of features passed in the parameter.
        '''
        self.linear = None
        # Initialize weights to zero for consistency with original implementation
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        '''
        TODOs: Implement forward pass with softmax/sigmoid activation function
        '''

        return None


class CustomAdamOptimizer:
    """
    YOUR Implementation of Adam optimizer
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        """
        Initialize custom Adam optimizer
        
        params -- Model parameters to optimize
        lr -- Learning rate (default: 1e-3)
        betas -- Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps -- Term added to denominator for numerical stability (default: 1e-8)
        weight_decay -- L2 regularization coefficient (default: 0.0)
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize state
        self.state = {}
        for param in self.params:
            self.state[param] = {
                'step': 0,
                'exp_avg': torch.zeros_like(param.data),      # m_t (first moment estimate)
                'exp_avg_sq': torch.zeros_like(param.data)    # v_t (second moment estimate)
            }
    
    def zero_grad(self):
        '''
        TODOs: Set gradients of all parameters to zero
        '''
        pass
    
    def step(self):
        '''
        TODOs: Perform a single optimization using Adam
        '''
        for param in self.params:
            if param.grad is None:
                continue
            
            grad = param.grad.data
            state = self.state[param]
            
            ''' 
            TODOs: Add weight decay (L2 regularization)
            '''
            grad = None

            # Get state variables
            exp_avg = state['exp_avg']      # m_t
            exp_avg_sq = state['exp_avg_sq'] # v_t
            
            # Increment step counter
            state['step'] += 1
            step = state['step']
            
            '''
            TODOs: 
            
            1. Update biased first moment estimate: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t

            2. Update biased second raw moment estimate: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²

            3. Compute bias-corrected first moment estimate: m̂_t = m_t / (1 - β₁ᵗ)

            4. Compute bias-corrected second raw moment estimate: v̂_t = v_t / (1 - β₂ᵗ)

            5. Compute step size: α_t = α * √(1 - β₂ᵗ) / (1 - β₁ᵗ)

            6. Update parameters: θ_t = θ_{t-1} - α_t * m_t / (√v_t + ε)
            '''


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, json_line, vocab, use_bias=True):
        """
        Create a new example

        json_line -- The json object that contains the label ("label") and features as fields
        vocab -- The vocabulary to use as features (list)
        use_bias -- Include a bias feature (should be false with Pytorch since nn.Linear has bias)
        """
        
        self.nonzero = {}
        self.y = 1 if json_line["label"] else 0
        self.x = np.zeros(len(vocab))
        self.raw_features = {}  # Store raw features for DictVectorizer

        for feature in json_line:
            if feature != "label":  # Store all non-label features
                self.raw_features[feature] = float(json_line[feature])
            
            if feature in vocab:
                assert feature != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(feature)] += float(json_line[feature])
                self.nonzero[vocab.index(feature)] = feature
        
        # Note: PyTorch nn.Linear handles bias automatically
        if use_bias and kBIAS in vocab:
            bias_idx = vocab.index(kBIAS)
            self.x[bias_idx] = 1.0



class ToyLogisticBuzzer(Buzzer):
    """
    PyTorch logistic regression classifier with custom Adam optimizer implementation.
    """

    def __init__(self, num_features, mu=0.0, learning_rate=0.1, betas=(0.9, 0.999), eps=1e-8):
        """
        Initialize the PyTorch logistic regression model with custom Adam optimizer
        
        num_features -- The number of features in the model
        mu -- Regularization parameter (L2 penalty)
        learning_rate -- Learning rate for the optimizer
        betas -- Adam beta parameters (default: (0.9, 0.999))
        eps -- Adam epsilon parameter (default: 1e-8)
        """
        self._dimension = num_features
        self.mu = mu
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        
        # Initialize the PyTorch model
        self.model = SimpleLogreg(num_features)
        
        # Initialize the loss function (Binary Cross Entropy)
        self.criterion = nn.BCELoss()
        
        # Initialize our custom Adam optimizer
        self.optimizer = CustomAdamOptimizer(
            self.model.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=mu  # L2 regularization
        )
        
        # For compatibility with original implementation
        self.filename = "pytorch_model"
        self._features = []
        self._correct = []
        self._feature_generators = []
        self.vectorizer = None  # Will store the DictVectorizer if used

    def progress(self, examples):
        """
        Given a set of examples, compute the probability, accuracy,
        precision, and recall which is returned as a tuple.

        examples -- The dataset to score (can be Examples or numpy arrays)
        """
        
        self.model.eval()  # Set to evaluation mode
        logprob = 0.0
        tp = fp = fn = tn = 0
        
        with torch.no_grad():
            # Handle both Example objects and numpy arrays
            if isinstance(examples, tuple) and len(examples) == 2:
                # It's (X, y) numpy arrays
                X, y = examples
                for i in range(len(X)):
                    x_tensor = torch.FloatTensor(X[i]).unsqueeze(0)
                    p = self.model(x_tensor).item()
                    
                    if y[i] == 1:
                        logprob += log(p + 1e-10)
                    else:
                        logprob += log(1.0 - p + 1e-10)

                    # Get accuracy metrics
                    if p < 0.5 and y[i] < 0.5:
                        tn += 1
                    elif p < 0.5 and y[i] >= 0.5:
                        fn += 1
                    elif p >= 0.5 and y[i] >= 0.5:
                        tp += 1
                    else:
                        fp += 1
                total = len(X)
            else:
                # It's a list of Example objects
                for ex in examples:
                    x_tensor = torch.FloatTensor(ex.x).unsqueeze(0)
                    p = self.model(x_tensor).item()
                    
                    if ex.y == 1:
                        logprob += log(p + 1e-10)
                    else:
                        logprob += log(1.0 - p + 1e-10)

                    # Get accuracy metrics
                    if p < 0.5 and ex.y < 0.5:
                        tn += 1
                    elif p < 0.5 and ex.y >= 0.5:
                        fn += 1
                    elif p >= 0.5 and ex.y >= 0.5:
                        tp += 1
                    else:
                        fp += 1
                total = len(examples)

        return {"logprob": logprob,
                "acc":     (tp + tn) / total,
                "prec":    tp / (tp + fp + 0.00001),
                "recall":  tp / (fn + tp + 0.00001)}

    def train_with_sklearn_features(self, train_examples, test_examples=None, passes=1, batch_size=32):
        """
        Train using sklearn's DictVectorizer for feature extraction
        """
        # Use the sklearn feature extraction
        X_train, y_train, X_test, y_test, vectorizer = create_feature_matrix_sklearn(
            train_examples, test_examples
        )
        self.vectorizer = vectorizer
        
        # Update model dimension if needed
        if X_train.shape[1] != self._dimension:
            logging.info(f"Adjusting model dimension from {self._dimension} to {X_train.shape[1]}")
            self._dimension = X_train.shape[1]
            self.model = SimpleLogreg(self._dimension)
            self.optimizer = CustomAdamOptimizer(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.mu
            )
        
        # Create PyTorch datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        update_number = 0
        
        for epoch in range(passes):
            for batch_features, batch_labels in train_loader:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_features).squeeze()
                
                # Compute loss
                loss = self.criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                
                # Update weights using custom Adam
                self.optimizer.step()
                
                update_number += 1
                
                # Log progress periodically
                if update_number % 10 == 1:
                    train_progress = self.progress((X_train, y_train))
                    if X_test is not None:
                        test_progress = self.progress((X_test, y_test))
                    else:
                        test_progress = defaultdict(int)
                        test_progress['logprob'] = float("-inf")

                    message = "SKLearn Batch %6i\t" % update_number
                    for fold, progress in [("Train", train_progress), ("Dev", test_progress)]:
                        for stat in progress:
                            message += "%s%s = %0.3f\t" % (fold, stat, progress[stat])
                    logging.info(message)
        
        # Inspect features
        feature_names = vectorizer.get_feature_names_out()
        self.inspect_sklearn(feature_names)

    def inspect_sklearn(self, feature_names, limit=5):
        """
        Inspect top features when using sklearn vectorizer
        """
        weights = self.model.linear.weight.data.squeeze().numpy()
        
        # Get indices sorted by weight magnitude
        feature_weights = [(i, weights[i], feature_names[i]) for i in range(len(weights))]
        feature_weights.sort(key=lambda x: x[1], reverse=True)
        
        logging.info("\n=== Top and Bottom Features (sklearn) ===")
        for idx, weight, name in feature_weights[:limit]:
            logging.info("Top Feat %35s %3i: %+0.5f" % (name, idx, weight))
        
        for idx, weight, name in feature_weights[-limit:]:
            logging.info("Bottom Feat %35s %3i: %+0.5f" % (name, idx, weight))

    def get_optimizer_state(self):
        """
        Return a dict mapping 'param_0', 'param_1', … to a summary of the Adam
        state expected by the unit-tests:

            {
                'param_0': {
                    'step'            : int,
                    'exp_avg_norm'    : float,
                    'exp_avg_sq_norm' : float
                },
                'param_1': { … }
            }
        """
        state_info = {}
        for idx, param in enumerate(self.model.parameters()):           # param_0, param_1, …
            adam_state = self.optimizer.state[param]
            state_info[f'param_{idx}'] = {
                'step'            : adam_state['step'],
                # L2 norms of the running averages
                'exp_avg_norm'    : adam_state['exp_avg'].norm().item(),
                'exp_avg_sq_norm' : adam_state['exp_avg_sq'].norm().item()
            }
        return state_info

    def sg_update(self, train_example, iteration):
        """
        Perform a single stochastic gradient update using custom Adam
        """
        self.model.train()  # Set to training mode
        
        # Convert to tensors
        x_tensor = torch.FloatTensor(train_example.x).unsqueeze(0)
        y_tensor = torch.FloatTensor([train_example.y]).unsqueeze(0)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(x_tensor)

        # Compute loss
        loss = self.criterion(output, y_tensor)

        # Backward pass
        loss.backward()
        
        # Update weights using custom Adam
        self.optimizer.step()
        
        # Return current weights for compatibility
        weights = self.model.linear.weight.data.squeeze().tolist()
        bias = self.model.linear.bias.data.item()
        return np.array(weights + [bias])

    def finalize_lazy(self, iteration):
        """
        Finalization step (kept for compatibility)
        """
        params = []
        for param in self.model.parameters():
            params.extend(param.data.flatten().tolist())
        return np.array(params)

    def inspect(self, vocab, limit=5):
        """
        A function to find the top features.
        """
        weights = self.model.linear.weight.data.squeeze().numpy()
        
        feature_weights = [(i, weights[i]) for i in range(len(weights))]
        feature_weights.sort(key=lambda x: x[1], reverse=True)
        
        top = [idx for idx, weight in feature_weights[:limit]]
        bottom = [idx for idx, weight in feature_weights[-limit:]]
        
        for idx in list(top) + list(bottom):
            logging.info("Feat %35s %3i: %+0.5f" %
                        (vocab[idx], idx, weights[idx]))
        
        return top, bottom

    def train(self, train=None, test=None, vocab=None, passes=1, batch_size=32, use_sklearn=False):
        """
        Given a dataset, learn a weight vector for your classifier.

        train -- the dataset to learn from
        test -- the dataset to validate results on
        vocab -- the names of the features used
        passes -- how many epochs to train
        batch_size -- batch size for training
        use_sklearn -- whether to use sklearn's DictVectorizer for feature extraction
        """
        
        if use_sklearn:
            # Use the new sklearn-based training
            logging.info("Using sklearn DictVectorizer for feature extraction")
            self.train_with_sklearn_features(train, test, passes, batch_size)
        else:
            # Original training method
            if vocab and len(vocab) != self._dimension:
                logging.warn("Mismatch: vocab size is %s, but dimension is %i" % \
                             (len(vocab), self._dimension))

            if not train:
                train = []
                vocab = self._feature_generators
                features = [feature.name for feature in self._feature_generators]
                assert len(self._features) == len(self._correct)
                for x, y in zip(self._features, self._correct):
                    x["label"] = self._correct
                    train.append(Example(x, features))
            else:
                assert vocab is not None, \
                    "Vocab must be supplied if we don't generate"

            # Original single-example updates with custom Adam
            update_number = 0
            for pass_num in range(passes):
                for ii in train:
                    self.sg_update(ii, update_number)
                    update_number += 1                

                    if update_number % 100 == 1:
                        train_progress = self.progress(train)
                        if test:
                            test_progress = self.progress(test)
                        else:
                            test_progress = defaultdict(int)
                            test_progress['logprob'] = float("-inf")

                        message = "Update %6i\t" % update_number
                        for fold, progress in [("Train", train_progress),
                                               ("Dev", test_progress)]:
                            for stat in progress:
                                message += "%s%s = %0.3f\t" % (fold, stat, progress[stat])
                        logging.info(message)

            self.finalize_lazy(update_number)
            if vocab:
                self.inspect(vocab)

    def save(self):
        """
        Save the PyTorch model and custom optimizer state
        """
        Buzzer.save(self)
        torch.save(self.model.state_dict(), f"{self.filename}.model.pth")
        # Save vectorizer if it was used
        if self.vectorizer:
            with open(f"{self.filename}.vectorizer.pkl", 'wb') as f:
                pickle.dump(self.vectorizer, f)
        # Optionally save optimizer state for resuming training
        optimizer_state = {
            'lr': self.optimizer.lr,
            'betas': (self.optimizer.beta1, self.optimizer.beta2),
            'eps': self.optimizer.eps,
            'weight_decay': self.optimizer.weight_decay,
            'state': {id(param): state for param, state in self.optimizer.state.items()}
        }
        torch.save(optimizer_state, f"{self.filename}.optimizer.pth")

    def load(self):
        """
        Load the PyTorch model and custom optimizer state
        """
        Buzzer.load(self)
        self.model.load_state_dict(torch.load(f"{self.filename}.model.pth"))
        # Load vectorizer if it exists
        try:
            with open(f"{self.filename}.vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
        except FileNotFoundError:
            pass
        # Optionally load optimizer state
        try:
            optimizer_state = torch.load(f"{self.filename}.optimizer.pth")
            self.optimizer = CustomAdamOptimizer(
                self.model.parameters(),
                lr=optimizer_state['lr'],
                betas=optimizer_state['betas'],
                eps=optimizer_state['eps'],
                weight_decay=optimizer_state['weight_decay']
            )
        except FileNotFoundError:
            logging.warning("Optimizer state file not found, using fresh optimizer")


def read_dataset(filename, vocab, limit, use_bias=False):
    """
    Reads in a text dataset with a given vocabulary

    filename -- json lines file of the dataset
    vocab -- the names of the features
    limit -- how many examples to read
    use_bias -- whether to include bias term (False for PyTorch)
    """

    if use_bias:
        assert vocab[0] == kBIAS, \
            "First vocab word must be bias term (was %s)" % vocab[0]

    dataset = []
    num_examples = 0
    with open(filename) as infile:
        for line in infile:
            num_examples += 1
            ex = Example(json.loads(line), vocab, use_bias=use_bias)
            dataset.append(ex)

            if limit > 0 and num_examples >= limit:
                break

    # Shuffle the data so that we don't have order effects
    random.shuffle(dataset)

    return dataset


if __name__ == "__main__":
    import argparse    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--step", help="Initial SG step size",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--vocab", help="Vocabulary of all features",
                           type=str, default="../data/small_guess.vocab")
    argparser.add_argument("--train", help="Training set",
                           type=str, default="../data/small_guess.buzztrain.jsonl", required=False)
    argparser.add_argument('--regularization', type=float, default=0.0)
    argparser.add_argument('--learning_rate', type=float, default=0.1)
    argparser.add_argument('--beta1', type=float, default=0.9, help="Adam beta1 parameter")
    argparser.add_argument('--beta2', type=float, default=0.999, help="Adam beta2 parameter") 
    argparser.add_argument('--eps', type=float, default=1e-8, help="Adam epsilon parameter")
    argparser.add_argument("--limit", type=int, default=-1)
    argparser.add_argument("--test", help="Test set",
                           type=str, default="../data/small_guess.buzzdev.jsonl", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)
    argparser.add_argument("--batch_size", help="Batch size for training",
                           type=int, default=32, required=False)
    argparser.add_argument("--use_sklearn", help="Use sklearn DictVectorizer for features",
                           action='store_true', default=False)

    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    with open(args.vocab, 'r') as infile:
        vocab = [x.strip() for x in infile]
    print("Loaded %i items from vocab %s" % (len(vocab), args.vocab))
        
    # Note: use_bias=False for PyTorch since nn.Linear handles bias
    train = read_dataset(args.train, vocab=vocab, limit=args.limit, use_bias=False)
    test = read_dataset(args.test, vocab=vocab, limit=args.limit, use_bias=False)

    print("Read in %i train and %i test" % (len(train), len(test)))

    if args.use_sklearn:
        # When using sklearn, we let it determine the feature dimension
        _, _, _, _, temp_vectorizer = create_feature_matrix_sklearn(train[:1])
        num_features = len(temp_vectorizer.get_feature_names_out())
    else:
        num_features = len(vocab)

    # Initialize PyTorch model with custom Adam optimizer
    lr = ToyLogisticBuzzer(num_features,
                           mu=args.regularization,
                           learning_rate=args.learning_rate,
                           betas=(args.beta1, args.beta2),
                           eps=args.eps)

    # Train the model
    lr.train(train, test, vocab, args.passes, args.batch_size, use_sklearn=args.use_sklearn)

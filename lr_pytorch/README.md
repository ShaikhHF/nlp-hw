Logistic Regression Redux: Pytorch
=

Overview
--------

In this homework you'll implement a stochastic gradient ascent for logistic
regression and you'll apply it to the task of determining whether documents
are talking about hockey or baseball.  Sound familiar?  It should be!

Indeed, it will be doing exactly the same thing on exactly the same data as
the [Logistic Regression Homework](https://github.com/Pinafore/nlp-hw/tree/master/lr_sgd_qb).  The only difference is that while you had to do logistic regression yourself, this time you'll be able to use Pytorch directly.

What you have to do
----

Coding (15 points)- `pytorch_custom_adam_buzzer.py`:

1. Load in the data and create a data iterator. You may use the sklearn feature creation functions to create a data vectorizer or you can do it yourself to directly create a matrix. Modify the `create_feature_matrix_sklearn()` function.
1. Create a logistic regression model with a softmax/sigmoid activation function.  To make unit tests work, we had to initialize a member of the SimpleLogreg class. Replace the none object with an appropriate nn.Module in the `SimpleLogreg` class. 
1. Optimize the Adam optimizer function (remember to zero out gradients) in the `CustomAdamOptimizer` class.

Analysis (5 points):

1. How does the setup differ from the model you learned "by hand" (Logistic regression homework) in terms of initialization, number of parameters, activation?
2. Look at the top features again.  Do they look consistent with your results for the last homework? **Hint**: run `pytorch_custom_adam_buzzer.py` and `toylogistic_buzzer.py` to compare the magnitude of the top features.


Simple tests:
--
Run `test_custom_adam.py`

```........
Before any update:
  A     +0.000000
  B     +0.000000
  C     +0.000000
  D     +0.000000
  bias  +0.000000

kPOS.x: [1.0, 4.0, 3.0, 1.0, 0.0]

After first update:
  A     +0.100000
  B     +0.100000
  C     +0.100000
  D     +0.000000
  bias  +0.100000

kNEG.x: [1.0, 0.0, 1.0, 3.0, 4.0]

After second update:
  A     +0.167006
  B     +0.133351
  C     +0.045439
  D     -0.074414
  bias  +0.083923
.
----------------------------------------------------------------------
Ran 9 tests in 0.491s

OK
```

What to turn in
-

1. Submit your `pytorch_custom_adam_buzzer.py` file (include your name at the top of the source)
1. Submit your _analysis.pdf_ file
    - no more than one page
    - pictures are better than text
    - include your name at the top of the PDF

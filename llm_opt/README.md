
Big Picture
=====

While we've trained models from scratch, often the best place to start is with
an existing model.  In this homework, we're going to do two things: 1) use
prompting to improve the outputs of a black box model for the guessing task 2)
fine tune a large(ish) open weight model so that it can do the same take that
you saw in your feature engineering homework: predicting whether an answer to
a question is correct or not.

This homework is worth 35 points, but distributed over three people.

General Setup
=====

As usual, you may need to install athe required Python environment (there's a lot this time).

    ./venv/bin/pip3 install -r requirements.txt

We'll also be using ollama:

     curl -fsSL https://ollama.com/install.sh | sh

You may need to install as ``sudo``.  After installing ollama, you'll want to
download the model you'll be using.  ``gemma3:4b`` will run on most hardware.
Things will be much better if you have a GPU to use.  The TAs will be making
Nexus accounts available if you do not have your own hardware.

What you Have to Do (Guesser)
=====

To get things set up, you'll need to train a **TfIdfGuesser**.  This was the
subject of a previous homework, so just use your previous solution for that.
The quality of this element matters a lot for the RAG step, so you may want to
spend some time tuning it if you didn't do it for the previous homework.
Recall is much more important than precision.

	.venv/bin/python3 guesser.py --guesser_type=Tfidf --question_source=gzjson --questions=data/qanta.guesstrain.json.gz --logging_file=guesser.log

The first priority is to get it running and sending requests to Ollama.  The
actual optimization process can take a while (it depends on your hardware), so
don't leave this until the last minute.  You also cannot request a Nexus
account later than a week before the due date, so if you need one, make sure
to ask ASAP.

Some suggestions on what you could do (non-exhaustive, and you can try other stuff):
- Change the pipeline to explicitly determine the lexical answer type
- Run multiple optimizations to separately tune the query formulation, guessing, and confidence estimation [if you do this, make sure you don't overfit on the validation data ... make sure to divide it up]
- Add more RAG inputs (e.g., Wikipedia)
- Tune the RAG outputs (right now, it's just the retrieved sentence, more context might be helpful)
- Create explicit intermediate results that could help calibration (e.g., number of RAG hits that match, overlap between RAG and output, does the guess appear in the question text)


What you Have to Do (Buzzer)
=====

After understanding the code, you can get down to coding:

* To make sure that this is actually efficient, you will need to freeze the
  model's original parameters.  Set the `requires_grad` for all of the base
  model parameters to `False`.  You need to do this in the
  `initialize_base_model` code.  *Do not overlook this, as it will work
  without this change but will be very slow*.  The first time you run the code will take a little bit longer because it needs
to download the DistillBERT model.

      ./venv/bin/python3 lorabert_buzzer.py 
      config.json: 100%|████████████████████████████████████████████| 483/483 [00:00<00:00, 7.18MB/s]
      model.safetensors: 100%|████████████████████████████████████| 268M/268M [00:04<00:00, 64.1MB/s]

This will go faster afterward.


* You will need to define the parameter matrices for the LoRA layer in the
  `LoRALayer` class `__init__` function and then use them to compute a delta
  in the `forward` function.

* Likewise, you will need to add a `LoRALayer` component to the `LinearLoRA`
  class and change its `forward` function to use that delta in its forward
  function.  (I realize this could have been one class, but this makes testing
  easier... it also makes it possible to have more LoRA adaptations beyond
  adapting just linear layers.)

* Now that we have the tools for changing some layers, we now need to add them
  to the frozen model we created in `initialize_base_model` in the `add_lora`
  function.  You will probably want to create a (partial
  object)[https://docs.python.org/3/library/functools.html#partial-objects].  

* You shouldn't need to change anything in `LoRABertBuzzer`, it should run the
  training for you and prepare the data.

* Run adaptation on some data (use `limit` if you don't have a GPU).  This is
  more of a proof-of-concept, and you don't need great accuracy to satisfy the
  requirements of the homework (but loss should go down and accuracy should
  improve with more data).


Good Enough Solution
=====

To have a good enough solution, you must both
1. improve the Guesser to have a higher recall and precision over the working implementation you've been given **conditioned on the underlying black box Muppet Model**.
2. improve the expected wins of the finetuned over either logistic regression with just the confidence feature and length or logistic regression with just the confidence feature and the guess.  (This will depend on your Guesser, obviously)

What to Submit
=====

* Your `analysis.pdf` file (if you don't go beyond the "Good
Enough", you must at least establish your baseline values).

* Your `lorabert.model` file (where you did your finetuning).

* Your `dspy.json` model (the final prompts found via teleprompting).

Extra Credit
======

* [Up to 10 Points] Improve the performance of the overall system.  We already
  talked about the Guesser, but for the Buzzer there are vey easy ways to do
  this: we are forming the `text` field of the examples in a fairly naive way.
  We could add more information or format it better.  A more involved (but
  likely better) is to further extend the model to better encode additional
  floating point features (like you did in the feature engineering homework).

* [Up to 5 Points] Experiment with what layers are most necessary for the best
  improvements and test values of alpha and rank that work best (you cannot
  use tiny datasets for this, unfortunately, so this requires a GPU, probably
  ... not a great big one, as any GPU will likely be fine).  Make sure in
  addition to any accuracy / buzz ratio numbers you provide you also count the
  number of parameters.

* [Up to 3 Points] The training code in `train` are taken directly from the
  Huggingface examples and I didn't think too much about them.  It's not clear
  that they're a good fit for the data.  Can you find something substantially
  better?  (Keeping the model / adaptation / etc. constant.) 


FAQ
========

*Q: Why do the unit tests use "encoder" but the template code uses "transformer"?*

*A:* The unit tests are using a "real" (but very tiny) BERT, while the template code is using DistilBERT (but much larger).  They are packaged slightly differently.

*Q: What Ollama models can I use?*

*A:* You may use: Gemma3:4b.  [Updated 27. October 2025, we'll likely add more]

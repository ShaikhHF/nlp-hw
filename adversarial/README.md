
This homework is about writing questions that are interesting and that
a reasonable human could answer but that computers struggle to answer.

Tools to Write Your Question
=======

Here are some techniques to write questions systems cannot answer:
https://www.youtube.com/watch?v=6oZCIOBiSaI

You can see a [previous competition we ran](youtube.com/watch?list=PLegWUnz91WfsBdgqm4wrwdgtPV-QsndlO&si=QulAvORs30rHWr4c&embeds_referring_euri=https%3A%2F%2F1227273660-atari-embeds.googleusercontent.com%2F&source_ve_path=Mjg2NjQsMTY0NTA2&v=5sYXzNE07nM&feature=youtu.be).

You are free to download/use any QA systems you'd like or to use them on the web.  Some ideas include:
* https://you.com/
* https://www.bing.com/new
* https://chat.openai.com/
* https://www.llama.com/
* https://deepmind.google/models/gemma/
* https://aistudio.google.com/

Please share any good ones you find with the rest of the class.

Format of the Question
========

You'll create two kinds of questions:
 * One will be pyramidal, since this is the format we've been using for all of the course projects.
 * The other will be multiple choice (MC), which is the format we'll use for the final exam.

You are welcome to share information between the two (i.e., you can turn your
multiple choice question into a pyramidal question).

Some resources for writing questions:
https://www.naqt.com/resources/question-writing.html

There are many reasons why MCQA isn't great, but everybody (especially us) uses it because it's easy to grade.

How to describe your question
=========
A question requires about a page of explanation with the following subsections:

* _Question_: The question you asked, why you thought the question is difficult for computers while not as hard for the humans, why you chose this specific topic to ask a question about. If it is pyramidial, why choose to do so?
* _Guess_: What answer systems provided (make sure to provide details on what systems you tried against). The details may include why you chose this answer system.
* _Answer_: What the correct answer should be.  Provide sources with citations.  Wikipedia should not be used as a primary source.  Better sources are peer-reviewed articles (JSTOR is your friend and is available from UMD IPs without a fee), newspapers, books (Google books is good), etc.
* _Guess Explanation_: Why did the AI provide the answer that it did? Connect it with why you chose this particular answer system and if the guesss it gave matches with your intent of making the question difficult. For the
homework, this can be in general terms (but do make use of concepts like
semantic/syntactic ambiguity that we've covered in class), but for the project
this should be backed up with relevant citations from the NLP literature.
* _Interesting/Notability_: Why would someone find this question interesting
or why would someone want to know the answer to this question?  Use citations
as relevant/necessary.  Possible explanations are: surprisingness (Brian May
from Queen has a PhD), novelty (John Tyler is the earliest former
President of the US with a living grandchild), importance (the Witting
reaction is taught in most organic chemistry classes), or connective (Samuel
Beckett drove Andre the Giant to School).

There's an example of this in this directory.  The citations do not count against the page limit, and it's okay to go slightly beyond a page.

Use of ChatGPT
========

Just so there are no mixed signals, you're allowed (and encouraged) to use
ChatGPT and other AI tools to check your work.  However, don't try to use them
to generate questions or explanations (it probably won't work).  In general,
adhere to the policy articulated here:
https://2023.aclweb.org/blog/ACL-2023-policy/

What Can you Write Questions About?
=========

You can write a question about anything.  We care more about the quality of
the question than the topic.  If it tests a skill that humans have that
computers lack, that's fine.  That said, your classmates will try to answer
these questions, so don't make it too obscene, difficult, obscure, or
annoying.  You will get more points for questions that are fun and a pleasure
to answer.

If you're at a loss, we strongly encourage you to write a question about the
contents of the course.  That's something you now know a lot about (as does
everyone in the class), so that will be a good topic for the final exam.

Groups
=========

You can work in groups for this assignment, but it doesn't decrease the
workload (e.g., if you're N people, you must submit N pyramical questions and N multiple choice questions).  Everyone
will get the same score for their submission, and if late days are used it
will be applied to all members of the group.

Question Security
==========

Do not share your questions with members of the Maryland Academic Quiz Team
(present or past) who are not members of the class without explicit permission
(they---along with course staff---will be our humans evaluating the quality of the questions).

What to turn in
==========

Turn in both a PDF of the writeup of your question and a json including the
raw text of the question and answer to Gradescope. Both examples of both
writeup and the json file are included in this directory.

Please make sure that your JSON file passes a validator (`python question_validator.py yourfile.json`) before submitting. This file check whether JSON is in correct format and your questions (either pyramidal or multiple choice) contains text, answer, and choices (if MC).

There should be one tossup and one bonus for each member of the group.

How do I know if I have a Good Question?
==========

You should make sure that your teammates / roommates / family can answer the
question (without knowing the answer).  You only get one chance to test a
question for the first time, so don't waste it.

Grading
==========

You will lose points if your question is:
* Vague
* Has incorrect information
* Has incorrect grammar
* Makes it seem like it's asking for two different things
* Is a near copy of an existing question

You will lose points if your answer is:
* Too obscure for anyone to know

Good questions will get full points if:
* The question is difficult with an accessible answer
* Clue-dense
* Reward knowledge
* Have specific facts
* Stump computers
* Are answerable by any reasonably-informed humans

Writeups will get full points if they:
* Cite all facts in the question with good (i.e., not Wikipedia) references
* Clearly explain why they structured the information and the pyramidality

The grading for the assignment will have multiple components:
*   The difference between human and computer difficulty (we will run on a selection of models and with human subjects)
*   A qualitative examination of how well-written the question is
*   Factual accuracy
*   The insight / completeness of the writeup

FAQ
===========

*Q:* How many total questions do I need to submit?

*A:* Each person is responsible for writing one pyramidal question (tossup) and one multiple choice question.  So if you have a group of three, that's three pyramidal questions and three multiple choice questions.  

*Q:* Can I submit more questions?

*A:* No, submit your best questions.  However, you will probably need to write many bad questions before you manage to write a good question.

*Q:* How strict is the one page limit?

*A:* It's fine to go beyond 1 page per question with those LLM
responses as long as your explanation doesn't. 

*Q:* Is it okay if the LLM doesn't answer the question because of
safety triggers?

*A:* Yes, that’s a fine strategy for this. But obviously your
questions should actually be safe for work.

*Q:* What humans should be able to answer the question?  Is it okay if it's hard?

*A:* At the minimum, a skilled trivia player should be able to get the question right.  However, it's even better if more people can get the question right.  The easier the question is for an average human (let's assume American undergrad for the purposes of this exercise), the more impressive it is that a computer cannot answer it.

*Q:* Do I have to use a Wikipedia page title as the answer?

*A:* Normally yes.  It helps to standardize and will also let us link it to resources.  However, some of the answers are not going to be in that set.  E.g., if you want to have an answer like "missing a leg" or "ways Sean Bean has died in films" or "because they're all dead", that's not going to match a page.  Please try to match if you can, but otherwise, an arbitrary string is fine.

*Q:* Can the answer be a year or a number?

*A:* Yes, but these questions are often difficult to write so that they're uniquely identifying.  For events that are ancient history, there's often debate about when something happened.  And there are different calendar systems.  You you'd need to specify something like "Based on the Greogorian calendar".  You also need to be specific about if you want a year, a day, a month, etc.  There's also sometimes confusion about when something happened: battles can last days, elections are voted on in November but someone isn't sworn in until the following January.  So all of this is to say, it's okay to ask this, but it sometimes requires more care.

Also, humans don't memorize a lot of dates.  Some things are tightly tied to dates (coronation of Charlemagne, Pearl Harbor, September 11), but most things are not.  So if we're looking for things that humans can answer but computers cannot, these sorts of things may be more difficult.

*Q:* For an MCQ, what if the system gets the right answer for the
wrong reason?

*A:* I agree that this is
 (https://arxiv.org/abs/2502.14127)[suboptimal].  However, it's how
 MCQA works: accuracy is all that's measured.  Take a look at the
 reasoning (or your distractor answers) to see how you can get it to
 go for something that wouldn't fool a human.

*Q:* What are considered valid citations?  What if I'm writing a
 question about video games?  Or what if I want to ask a question
 about what something looks like?

*A:*: For things like this, fan-sourced wikis are usually the best /
 only option.  You can also use a well-sourced photograph.

*Q*: You say that the human skill will be computed with respect to
 ``an average UMD student''.  Is it okay if it's something only a UMD
 student would know?

*A:* UMD-specific information is okay, but grad students and
 professors should know what you're asking about.  Otherwise, we won't
 be able to judge whether it is actually well known.

*Q:* What information should we give about the AI answers?

*A:* You only need to provide the AI answers.  If you want to provide
 snippets of the reasoning (because it's informative or just funny),
 that's fine too.  Often, your analysis of what the AI did will be
 more valuable than its (potentially faulty) introspection.

*Q:* Do all clues in a pyramidal question need to be uniquely
identifying?

Let's consider what the definition of pyramidal is: the question gets easier and easier such that smarter subjects can answer the question earlier.  In an extreme case, if the question is only answerable at the very end of the question, then it cannot be pyramidal.   So, yes, the first sentence needs to be uniquely identifying.  However, it is okay for subsequent sentences / clue to be less identifying.  That being said, computers often get tripped up when you explicitly exclude the ambiguity.  E.g.:

```


It's not headquartered in Biel, Switzerland, but its logo looks like the last letter of the Greek alphabet.
It's not the protagonist of Thirty Rock, but ...
```

*Q:* How can I make a spatial reasoning question pyramidal?

*A:* To make a geometry question pyramidal, you can walk through the steps to solve the problem. In other words, provide the correct chain of thought that the model would use to solve the problem.

```

This point is the center of the ellipse that passes through (0, 1), (2, 0), (3/2, \sqrt{3} + 2), (3/2, \sqrt{3} - 2), and (2, 2).  To find the center, the first step would be to recognize that the line connecting (2, 2) and (2, 0) must pass through the center.  Next, you can recognize that (2, 0) must lie on the semi-minor axis but that (0, 1) must lie on the semi-major axis.  This implies that the semi-major axis is 4 but the semi-minor is 2, so the radius must be one.  This puts the center of the ellipse at the midpoint of (2, 2) and (2, 0), which---for 10 points---is what point?
```

*Q:* Is it okay of MCQ questions are identical to the pyramidal
questions?

It's probably suboptimal for them to be **identical**.  First, a good pyramidal question when viewed as a one-shot MCQ question will have some redundancy:

```

Pyramidal: He wrote about how the incompetency of Franco's marksmen saved his life while fighting in the Spanish Civil War.  He wrote about his inability to earn a living wage in two European capitals.  In a fictional book, some of his characters proclaimed that "two legs are better", while another fictional book featured the "two minutes of hate".  For 10 points, name this author born Eric Arthur Blair who wrote Homage to Catalonia, Down and Out in London and Paris, Animal Farm, and 1984.
```

If a system sees all of that at once, the only relevant portion could be reduced to:

```

Who wrote Homage to Catalonia, Down and Out in London and Paris, Animal Farm, and 1984?
```

*Q:* What computers should ~not~ be able to answer the questions?

*A:* It's okay if some computers can answer the questions.  But clearly it would be better if all computers cannot answer the question.  Even better is if they all fail in different ways.  We're not going to focus on particular systems and say that, for instance, it must absolutely stump ChatGPT.

Unfortunately, this is not so easy to quantify, as not all systems are equal.  It's obviously better to defeat "better" systems.  And if it's something that every human gets right, it's okay if fewer computers are fooled.  We care about the gap between human and computer difficulty, not the absolute computer difficulty.  

*Q:* What if a system *sometimes* gets it wrong?

*A:* Obviously consistently wrong is better, but if there's sufficient stochasticity that it rarely gets it right, that's okay.

*Q:* What if only people with a particular background can answer the
question?

*A:* There are shades of this.  For a pyramidal question, if only Koreans can answer the question at the start but everyone can at the end, that's okay (encouraged even!).  But if only Koreans could possibly answer the question, then that will read as if it's very difficult for the "average" human (as only 1% of humanity is Korean, and I think that percentage is also consistent with UMD student population).

*Q*: Do I have to use BibTeX for citations?

*A:* No, but it is a good skill to have.  As long as we can track down what you're talking about, it's okay.

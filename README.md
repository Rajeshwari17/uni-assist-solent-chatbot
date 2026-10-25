# Uni-Assist (Solent University prototype)

A small intent-classification chatbot that answers common questions about Solent University (library, sports, applying, fees, accommodation, wellbeing support, careers). It is a learning prototype and a study of where such systems fail. It is not deployed anywhere.

## How it works
- Questions are cleaned, then turned into numbers with TF-IDF on character and word n-grams (the character n-grams make it tolerant of typos).
- A logistic regression classifier predicts which topic ("intent") the question belongs to and gives a confidence score.
- If confidence is below a threshold, the bot says it is unsure and points to the Student Hub instead of guessing.
- Wellbeing and harassment topics use a lower threshold and fixed, safe replies, because missing a distressed student is worse than a false alarm.

## Run it
```
pip install scikit-learn numpy
python uniassist.py     # chat in the terminal
python evaluate.py      # evaluation on held-out questions
```

## Files
- `uniassist.py`: model and chatbot
- `intents.json`: training patterns and Solent responses
- `test_questions.json`: held-out paraphrase, typo and out-of-scope questions (never used for training)
- `evaluate.py`: accuracy, threshold sweep, sensitive-routing and out-of-scope checks
- `SOURCES.md`: where each fact in the responses came from (checked 24 October 2026)

## Results (small test sets: 32 paraphrases, 18 typos, 20 out-of-scope)
- Paraphrased questions: 78.1% correct (raw prediction, no threshold)
- Questions with typos: 94.4% correct
- Wellbeing and harassment questions routed to a safe reply: 6 of 6
- Unrelated questions wrongly answered at the shipped threshold: 4 of 20

These sets are small, so one question changes a score by about 3 points. Treat the numbers as indicative.

## What I learned
- The bot fails on vocabulary missing from the training patterns (for example the word "dorms").
- I added three accommodation patterns to fix that. It fixed the question I tested, but paraphrase accuracy fell from 81.2% to 78.1%, because new patterns pulled a fees question towards accommodation. Data changes need re-evaluation each time.
- A test set only measures what it contains: none of its questions measured the dorms fix.
- Confidence scores are not fully reliable. Some unrelated questions, such as "translate good morning into Spanish", were answered as greetings with high confidence.

## Limitations
- No conversation memory, and only a closed set of topics.
- Training patterns and test questions were written by an AI assistant, not collected from real students. They are probably easier than real queries.
- Not tested with real users.
- The sensitive-topic routing is a simple classifier, not a safeguarding system. A real deployment would need review by wellbeing staff.
- Responses link to official Solent pages instead of stating prices, hours and deadlines, which change.

## Credits and AI assistance
This project was built with an AI assistant (Claude, by Anthropic), which wrote the code and the first datasets. My own contribution: I ran and tested the bot, edited the training data, ran the evaluations, and interpreted the results. It is separate from, and later than, my COM727 group chatbot project at Solent University, where my role was testing.

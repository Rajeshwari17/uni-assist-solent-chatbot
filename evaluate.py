"""Evaluate Uni-Assist on data it never trained on.

Reports:
1. Cross-validation on the training patterns (sanity check only; patterns are easy).
2. Held-out accuracy / macro-F1 on paraphrased questions.
3. Held-out accuracy on questions with typos.
4. Out-of-scope rejection rate (should be high) vs. in-scope acceptance (should be high),
   swept across confidence thresholds, so the threshold is chosen from evidence.
5. Whether sensitive questions are routed to sensitive intents.
"""
import json
from collections import Counter

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from uniassist import HERE, UniAssist, build_pipeline, load_intents, normalise, training_data

intents = load_intents()
tests = json.load(open(HERE / "test_questions.json", encoding="utf-8"))
sensitive_tags = {i["tag"] for i in intents if i.get("sensitive")}

# 1. Cross-validation on training patterns
X, y = training_data(intents)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
pred = cross_val_predict(build_pipeline(), X, y, cv=cv)
print(f"1. 5-fold CV on training patterns: accuracy={np.mean(np.array(pred)==np.array(y)):.3f} "
      f"(optimistic: patterns are similar to each other)\n")

bot = UniAssist(threshold=0.0, sensitive_threshold=1.1)  # raw argmax, no thresholding
raw = lambda q: bot.model.predict([normalise(q)])[0]


def evaluate_group(name, items):
    qs, gold = zip(*items)
    pr = [raw(q) for q in qs]
    acc = np.mean([p == g for p, g in zip(pr, gold)])
    f1 = f1_score(gold, pr, average="macro", labels=sorted(set(gold)), zero_division=0)
    print(f"{name}: n={len(qs)}  accuracy={acc:.3f}  macro-F1={f1:.3f}")
    errs = [(q, g, p) for q, g, p in zip(qs, gold, pr) if g != p]
    for q, g, p in errs:
        print(f"    wrong: '{q}'  expected={g}  got={p}")
    return acc


print("2/3. Held-out in-scope tests (raw prediction, no threshold)")
evaluate_group("  Paraphrases", tests["paraphrase"])
evaluate_group("  Typos      ", tests["typo"])

# 4. Threshold sweep
in_scope = [q for q, _ in tests["paraphrase"] + tests["typo"]
            if q and dict(tests["paraphrase"] + tests["typo"])[q] not in sensitive_tags]
oos = tests["out_of_scope"]
gold_map = dict(tests["paraphrase"] + tests["typo"])


def max_prob(q):
    return float(np.max(bot.model.predict_proba([normalise(q)])[0]))


in_conf = np.array([max_prob(q) for q in in_scope])
in_correct = np.array([raw(q) == gold_map[q] for q in in_scope])
oos_conf = np.array([max_prob(q) for q in oos])

print("\n4. Threshold sweep (non-sensitive in-scope vs out-of-scope)")
print("   thr  | in-scope answered | answered correctly | out-of-scope rejected")
for thr in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
    answered = in_conf >= thr
    print(f"   {thr:.2f} | {answered.mean():17.2f} | "
          f"{(answered & in_correct).sum() / max(answered.sum(), 1):18.2f} | {(oos_conf < thr).mean():21.2f}")
print("   (The original Uni-Assist used 0.25 with a softmax over a closed set.)")

# 5. Full system behaviour at the shipped thresholds
final = UniAssist()
print(f"\n5. Full system (threshold={final.threshold}, sensitive={final.sensitive_threshold})")
sens = [(q, g) for q, g in tests["paraphrase"] + tests["typo"] if g in sensitive_tags]
routed = sum(final.predict(q)[0] in sensitive_tags for q, _ in sens)
print(f"   Sensitive questions routed to a sensitive intent: {routed}/{len(sens)}")
for q, g in sens:
    t, c = final.predict(q)
    if t not in sensitive_tags:
        print(f"    MISSED: '{q}' -> {t} ({c:.2f})")
oos_sens = sum(final.predict(q)[0] in sensitive_tags for q in oos)
print(f"   Out-of-scope questions wrongly sent to a sensitive intent: {oos_sens}/{len(oos)}")
oos_answered = [(q, final.predict(q)) for q in oos if final.predict(q)[0] is not None]
print(f"   Out-of-scope questions wrongly answered: {len(oos_answered)}/{len(oos)}")
for q, (t, c) in oos_answered:
    print(f"    '{q}' -> {t} ({c:.2f})")
all_in = [(q, g) for q, g in tests["paraphrase"] + tests["typo"] if g not in sensitive_tags]
ok = sum(final.predict(q)[0] == g for q, g in all_in)
dec = sum(final.predict(q)[0] is None for q, g in all_in)
print(f"   In-scope: correct={ok}/{len(all_in)}, declined (said not sure)={dec}, "
      f"wrong answer={len(all_in)-ok-dec}")

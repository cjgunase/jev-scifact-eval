# SciFact question development log

Development used only the fixed 100-claim train sample (seed 20260929). Dev was not touched.

## v1 (frozen)

**Train run (2026-09-29):** 3,000 re-rank calls and 310 verify pairs (300 top-3 + 10 oracle). Cumulative spend after the run: $0.1158.

**Retrieval (train):**

| | R@1 | R@3 | R@10 |
|---|---|---|---|
| BM25 | 0.586 | 0.828 | 0.914 |
| Jev re-rank | 0.690 | 0.828 | 0.931 |

**Thresholds selected** (maximising abstract Label+Rationale F1 over a 0.05 grid): `tau_verdict = 0.8`, `tau_sentence = 0.9`.
Train official F1: abstract label-only 0.530, abstract rationalized 0.530, sentence selection 0.369, sentence label 0.363.

**Verdict confusion, all judged pairs (gold → predicted):**

| gold \ pred | supports | contradicts | not_enough_info |
|---|---|---|---|
| supports | 35 | 3 | 6 |
| contradicts | 0 | 12 | 2 |
| not_enough_info (non-gold abstract) | 38 | 28 | 186 |

On gold pairs, 47/58 (81%) of verdicts are correct.

**Illustrative errors:**
1. Claim 79, "Active caspase-11 protein promotes pyroptosis.": gold SUPPORT, predicted NEI at confidence 0.99. A confidently wrong case.
2. Claim 342, "Diabetic patients with acute coronary syndrome experience decreased … risk for bleeding": gold CONTRADICT, predicted NEI at 0.96.
3. Claim 584, "In rhesus macaques, daily subcutaneous injections of tenofovir protects …": gold SUPPORT, predicted CONTRADICT at 0.12 (low confidence, so filtered out by `tau_verdict`).
4. Claim 576, "In melanoma, anti-CTLA-4 treatment reinvigorates exhausted PD-1+Eomes+CD8 T cells.": gold SUPPORT, predicted CONTRADICT at 0.35 (filtered out).
5. Claim 726, "Ly6C hi monocytes have a higher inflammatory capacity than Ly6C lo monocytes.": gold SUPPORT, predicted NEI at 0.48.

**Decision: no revision.** The pre-registered revision trigger (gold CONTRADICT answered NEI in more than 30% of pairs) was not met: 2/14 = 14%.
The dominant error is supports/contradicts verdicts on abstracts outside the gold set (66/252). Part of this is a
property of SciFact: only abstracts cited for a claim were annotated, so an uncited abstract that bears on the
claim scores as a false positive. The `tau_verdict = 0.8` cut-off controls this without rewording. v1 is frozen
(tag `questions-frozen`).

# Sample-metadata annotation with Jev (TypeSafe System One)

## Aim
Test whether Jev can map free-text, GEO-style sample records to controlled
vocabularies, and whether its reported confidence flags the records that
a curator would also find hard.

## Data
`data/raw/geo_style_samples.json` holds 30 **synthetic** records written for this
test (not real GEO submissions). Each record has `title`, `source_name`,
`characteristics` and `protocol` free text, plus hand-assigned expected
labels and an `ambiguous` flag. Ten records were written to be ambiguous on purpose
(vehicle controls, organoids, iPSC derivatives, metastases, uninformative metadata).
The labels come from one annotator (the author), with no second rater, so
they are a reference point rather than ground truth.

## Model and request
- Model: `jev-latest` (resolved to `jev-1.13.0` on 2026-09-29), Python `typesafe-sdk` 0.7.2.
- One request per sample. State = `{"sample": {title, source_name, characteristics, protocol}}`;
  the expected labels are never included in the state.
- Five questions are evaluated in parallel within each request (see `scripts/annotate_samples.py`):
  - **Choice** `tissue` (9 options incl. `other_or_unclear`), `condition` (5 incl. `unclear`),
    `assay` (7 incl. `other`). Output: an argmax label, a probability distribution over options
    and a confidence derived from how concentrated that distribution is.
  - **Noul** `cell_line`, `treated`. Output: P(yes). A "yes" call uses P ≥ 0.5 (the neutral
    point, not tuned). For plotting, `|P − 0.5| × 2` serves as a confidence proxy.

## Evaluation
Accuracy of the argmax against the expected labels, per question, split by the ambiguous flag.
P(expected label) is also reported for each sample × question, which gives a
graded view of disagreement beyond the binary correct/incorrect call.

## Limitations
- n = 30 synthetic records from a single annotator. Accuracy estimates carry wide
  uncertainty (e.g. 30/30 gives a 95% Clopper–Pearson lower bound of about 0.88).
- Records were written by someone who knew the label set, so they are likely cleaner than real GEO metadata.
- Confidence was not calibrated against outcomes. With a single error, calibration cannot be assessed.
- Some label definitions are debatable (e.g. tissue of a lymph-node metastasis; the disease status of HEK293T cells).

## References
- TypeSafe documentation: https://docs.typesafe.ai (Choice, Noul, Confidence pages).
- Barrett T. et al. (2013) NCBI GEO: archive for functional genomics data sets—update. *Nucleic Acids Res* 41:D991–D995.

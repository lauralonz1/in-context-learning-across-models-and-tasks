# In-Context Learning Across Models and Tasks

Empirical study on the effect of in-context demonstrations across different language models and task types.

The project evaluates three models:

- Qwen3-30B-A3B-Instruct-2507
- Granite 4.0 H Micro
- Gemma-3-4B-it

on two benchmarks:

- MMLU-Pro
- QuALITY

The experiments study:

- the effect of the number of in-context demonstrations;
- random vs semantic demonstration selection;
- robustness to demonstration seed;
- context utilization on QuALITY.

## Project structure

```text
configs/
    experiment.yaml

data/
    splits/
        mmlu_pro_demo.jsonl
        mmlu_pro_full.jsonl
        mmlu_pro_sensitivity.jsonl
        quality_demo.jsonl
        quality_full.jsonl
        quality_sensitivity.jsonl

src/
    datasets/
    evaluation/
    models/
    prompts/
    selection/
    statistics/
    utils/

results/
    raw_predictions/
    statistics/

tests/
```

## Experimental setup

Model weights are frozen and no fine-tuning is performed.

The main decoding configuration is:

- temperature = 0
- top_p = 1
- max_new_tokens = 16

Random demonstration selection uses seed 42.

The seed-sensitivity analysis additionally evaluates seeds 43 and 44.

MMLU-Pro is evaluated with 0, 1, 3 and 5 demonstrations.

QuALITY is evaluated with 0, 1 and 3 demonstrations because longer few-shot configurations would exceed the common 32k-token context budget.

Demonstrations are selected either:

- randomly, without replacement;
- through semantic similarity with the target item.

The demonstration pools are disjoint from the evaluation items.

## Statistical analysis

For MMLU-Pro, paired comparisons are analysed using:

- exact McNemar tests;
- paired bootstrap 95% confidence intervals;
- Wilson 95% confidence intervals for individual accuracies;
- Holm correction for multiple comparisons.

For QuALITY, inference is performed at article level using:

- article-level sign-flip tests;
- cluster bootstrap 95% confidence intervals;
- Holm correction for multiple comparisons.

Seed-sensitivity results are treated descriptively.

## Installation

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

API credentials must be provided locally through a `.env` file.

The `.env` file is intentionally not included in the submission.

## Results

Final model predictions are stored in:

```text
results/raw_predictions/
```

Final statistical outputs are stored in:

```text
results/statistics/
```

The main result files are:

- `mmlu_main_inference.csv`
- `quality_main_inference.csv`
- `mmlu_seed_sensitivity.csv`
- `quality_seed_sensitivity.csv`
- `quality_context_ablation.csv`
- `key_findings_table_final.csv`
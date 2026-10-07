# Lab21 — LoRA fine-tuning

Nguyễn Vũ Anh · 2A202602502 · Track 3: AI Application

## Submission

- [Full report](submission/REPORT.md)
- [Results and verification log](results/)
- [Public HuggingFace adapter — bonus B5](https://huggingface.co/vustaz/Lab21-Qwen3.5-4B-LoRA)
- [Executed notebook](notebooks/Lab21_NguyenVuAnh_2A202602502.ipynb)
- [Submission links](LINKS.md)

Core NB1–NB5 completed with full evaluation (50 target, 15 regression). Final artifact verification: 119 tests passed, 26 checks passed, 1 warning, 0 failures.

## Honest experiment verdict

**FAIL**: target field accuracy improved from 0.765 to 0.980, but regression keyword recall fell from 0.791111 to 0.522222, beyond the 0.02 tolerance. A failed experiment verdict is gradeable under the lab rubric. No gate thresholds were changed.

Bonus B5 publishes the existing correct adapter; B1–B4 have not been performed. The sixth custom notebook cell packages the report and is not the optional NB6 merge/serve experiment.

## Reproduction

Source: [VinUni-AI20k/Day21-Track3-Finetuning-Lab](https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-Lab), commit `d27c1c02ebe99f32f52f706be88b4c30fb1d7fca`. Includes experiment patches and evidence.

Use T4 GPU, `COMPUTE_TIER=T4`, `LAB_MAX_LENGTH=256`; leave EVAL_LIMIT and EPOCHS unset (full eval, default 2 epochs). Install requirements with compatible CUDA torch, then run:

```bash
python scripts/colab_run.py nb1 nb2 nb3 nb4 nb5
python scripts/verify.py
```

The report records the completed run. After retraining, update the report from the new evidence before submission.

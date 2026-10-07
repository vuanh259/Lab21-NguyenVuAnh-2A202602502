import json, hashlib, pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd() / 'src'))
from labkit import generate, evaluate as ev, report
from labkit.config import get_tier
root = pathlib.Path.cwd()
frozen = json.loads((root/'results/baselines_frozen.json').read_text())
assert hashlib.sha256(generate.OPTIMIZED_PROMPT.encode()).hexdigest()[:16] == frozen['optimized_prompt_sha']
target = [json.loads(line) for line in (root/'data/eval_target.jsonl').read_text().splitlines() if line.strip()]
model, tok = generate.load_base(get_tier('T4'))
preds, _ = generate.generate_batch(model, tok, [r['input'] for r in target], system=generate.OPTIMIZED_PROMPT, label='report-only frozen baseline examples')
rows = [{'i':i, 'ticket':r['input'], 'label':r['label'], 'prediction':p, 'baseline_score':ev.triage_field_accuracy(p,r['label'])} for i,(r,p) in enumerate(zip(target,preds))]
report.write_json(rows,'predictions_baseline_b.json',results_dir=root/'results')
print('Report-only baseline replay target:',sum(r['baseline_score'] for r in rows)/len(rows))
print('Original frozen aggregate remains unchanged:',frozen['baseline_b']['target'])
regression = [json.loads(line) for line in (root/'data/eval_regression.jsonl').read_text().splitlines() if line.strip()]
rpreds, _ = generate.generate_batch(model, tok, [r['instruction'] for r in regression], system=None, max_new_tokens=96, label='report-only frozen regression examples')
rrows = [{'i':i, 'instruction':r['instruction'], 'keywords':r['keywords'], 'prediction':p, 'baseline_score':ev.keyword_recall(p,r['keywords'])} for i,(r,p) in enumerate(zip(regression,rpreds))]
report.write_json(rrows,'predictions_regression_baseline_b.json',results_dir=root/'results')
print('Report-only baseline replay regression:',sum(r['baseline_score'] for r in rrows)/len(rrows))

from pathlib import Path
import re, sys, subprocess
chosen = int(sys.argv[1])
assert chosen > 0
cfg = Path('src/labkit/config.py')
s, n = re.subn(r'(name="T4",[\s\S]*?max_length=)\d+', lambda m: m[1]+'1024', cfg.read_text(), count=1)
assert n == 1
cfg.write_text(s)
needle = 'TIER = get_tier(os.environ.get("COMPUTE_TIER", "T4"))'
extension = '\n# Experiment-only length override; keep hardware-tier defaults intact.\nif os.environ.get("LAB_MAX_LENGTH"):\n    from dataclasses import replace\n    TIER = replace(TIER, max_length=int(os.environ["LAB_MAX_LENGTH"]))\n'
for p in Path('notebooks').glob('*.py'):
 source=p.read_text()
 assert needle in source, p
 if '# Experiment-only length override;' not in source:
  p.write_text(source.replace(needle,needle+extension))
 compile(p.read_text(),str(p),'exec')
subprocess.run([sys.executable,'scripts/build_colab.py'],check=True)
print('Default T4 remains 1024. Set LAB_MAX_LENGTH='+str(chosen)+' for the measured experiment.')

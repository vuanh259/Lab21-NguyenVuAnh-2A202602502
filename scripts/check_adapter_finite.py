from pathlib import Path
import json, torch
from safetensors import safe_open
rows=[]
for folder in sorted(Path('adapters').iterdir()):
    for path in folder.glob('*.safetensors'):
        n=nonfinite=nonzero=0
        with safe_open(str(path), framework='pt', device='cpu') as f:
            for key in f.keys():
                tensor=f.get_tensor(key)
                n+=tensor.numel()
                nonfinite+=int((~torch.isfinite(tensor)).sum())
                nonzero+=int(torch.count_nonzero(tensor))
        rows.append({'run':folder.name,'file':str(path),'elements':n,'nonfinite_elements':nonfinite,'nonzero_elements':nonzero})
print(json.dumps(rows,indent=2))
Path('results/adapter_finite_check.json').write_text(json.dumps(rows,indent=2))
assert rows and all(row['nonfinite_elements']==0 for row in rows)

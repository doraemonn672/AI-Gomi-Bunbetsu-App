import csv, subprocess, sys
from pathlib import Path

settings=[(8,1e-3),(8,1e-2),(16,1e-3),(16,1e-2),(32,1e-3),(32,1e-2)]
Path('results').mkdir(exist_ok=True)
rows=[]
for ch,lr in settings:
    print(f"=== channels={ch}, lr={lr} ===")
    subprocess.run([sys.executable,'src/train.py','--channels',str(ch),'--lr',str(lr),'--epochs','5'],check=True)
    import json
    r=json.loads(Path('results/last_train.json').read_text(encoding='utf-8'))
    rows.append([ch,lr,r['test_accuracy']])
with open('results/parameter_search.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f); w.writerow(['channels','learning_rate','test_accuracy']); w.writerows(rows)
print('Saved results/parameter_search.csv')

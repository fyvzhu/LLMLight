"""Export one experiment's numeric JSONL fields to CSV; never combine algorithms."""
import argparse,csv,json
from pathlib import Path


def flatten(row,prefix=''):
    result={}
    for key,value in row.items():
        key=prefix+str(key)
        if isinstance(value,dict):result.update(flatten(value,key+'.'))
        elif value is None or isinstance(value,(str,int,float,bool)):result[key]=value
    return result


def export(root):
    outputs=[]
    for path in Path(root).rglob('metrics/*.jsonl'):
        rows=[flatten(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
        if not rows:continue
        target=path.with_suffix('.csv')
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with target.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        outputs.append(str(target))
    return outputs

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    print('\n'.join(export(p.parse_args().run_dir)))
